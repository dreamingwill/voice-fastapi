import sys
import os
from pathlib import Path
import re

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import Command, CommandSettings
from app.services.commands import CommandService, CommandCreatePayload, IntentClassifier

def setup_test_db():
    # Use in-memory SQLite for testing to avoid database pollution
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal

def test_intent_classifier():
    print("Testing IntentClassifier...")
    test_cases = [
        ("各号注意，站综合信息检查五分钟准备", True),
        ("今天天气不错", False),
        ("站综合信息检查停", True),
        ("起飞", True),
        ("我们要不要去吃饭", False),
        ("一分钟准备", True),
        ("分机参数下发", True),
        ("这里的环境很嘈杂", False),
        ("a", False), # Too short
        ("非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的话", False), # Too long
    ]
    
    for text, expected in test_cases:
        result = IntentClassifier.is_intent(text)
        print(f"Text: {text:40} | Expected: {expected} | Result: {result}")
        assert result == expected
    print("IntentClassifier tests passed!\n")

def test_rule_matching():
    print("Testing Rule-based Matching...")
    session_factory = setup_test_db()
    service = CommandService(session_factory=session_factory)
    
    # Initialize settings
    with session_factory() as db:
        db.add(CommandSettings(user_id=0, enable_matching=True, match_threshold=0.75))
        db.commit()

    commands = [
        CommandCreatePayload(text="各号注意，站综合信息检查五分钟准备", code="0001"),
        CommandCreatePayload(text="站综合信息检查一分钟准备", code="0002"),
        CommandCreatePayload(text="起飞", code="0003"),
        CommandCreatePayload(text="各号注意，对塔无线检查", code="0023"),
        CommandCreatePayload(text="分机参数下发", code="0037"),
        CommandCreatePayload(text="点火", code="0003_1"),
        CommandCreatePayload(text="发射", code="0003_2"),
    ]
    
    service.upload_commands(0, commands)
    
    match_cases = [
        ("个号注意，站综合信息检查五分钟准备", "各号注意，站综合信息检查五分钟准备", "exact"), 
        ("站综合信息检查庭", "站综合信息检查一分钟准备", "fuzzy"), 
        ("各号注意对塔无线检查", "各号注意，对塔无线检查", "rule"), 
        ("分机参数下放", "分机参数下发", "fuzzy"), 
        ("起废", "起飞", "fuzzy"), 
        ("法射", "发射", "fuzzy"),
        ("你好今天天气不错", None, None), # Should not match
    ]
    
    for text, expected_cmd, expected_type in match_cases:
        result = service.match_command(0, text)
        print(f"Input: {text:25} | Match: {str(result.command):40} | Type: {result.match_type} | Score: {result.score:.2f} | Intent: {result.intent_detected}")
        if expected_cmd:
            assert result.matched
            assert result.command == expected_cmd
            if expected_type:
                assert result.match_type == expected_type
        else:
            assert not result.matched

    print("Rule-based Matching tests passed!\n")

if __name__ == "__main__":
    try:
        test_intent_classifier()
        test_rule_matching()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
