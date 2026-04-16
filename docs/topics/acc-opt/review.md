Code Review: Uncommitted Changes on dev-6.1

  Overview

  This PR implements a two-stage command matching architecture with intent classification, addressing the customer's requirement to reduce false positives by distinguishing between command
   and non-command speech.

  Key Changes:
  1. Added IntentClassifier - rule-based intent detection
  2. Enhanced CommandMatchResult with intent_detected and match_type fields
  3. Implemented three-stage matching: exact → rule-based → fuzzy
  4. Expanded text aliases for common ASR errors
  5. Added configuration flag --command_intent_classification

  ---
  Code Quality & Style

  ✅ Strengths

  1. Clean Architecture: Well-structured three-stage matching approach aligns with the proposed design
  2. Configurable: Intent classification can be toggled via env variable or CLI arg
  3. Comprehensive Aliases: Good expansion of command_text_aliases.json to handle common ASR errors (e.g., "起废" → "起飞")
  4. Logging: Added informative log messages for debug purposes

  ⚠️ Issues to Address

  1. Regex Pattern Performance Concern
  # app/services/commands.py:419
  self._task_pattern = re.compile(
      r"(站综合信息检查|起飞信号检查|第一次综合检查|...)"
  )
  - Issue: The alternation regex has 10+ options and will be compiled on every CommandService instantiation
  - Recommendation: Consider using re.compile at module level or lazy initialization

  2. Inconsistent Pattern Matching Approach
  # app/services/commands.py:456
  task_match = self._task_pattern.search(stripped)
  - Issue: Using search() instead of match() - may match patterns in the middle of text
  - Recommendation: Use match() if patterns should only match from the start, or document why search() is intentional

  3. Magic Numbers Without Constants
  # app/services/commands.py:307
  if len(text) < 2 or len(text) > 40:
  - Issue: Hard-coded length thresholds
  - Recommendation: Extract to class constants with documentation:
  MIN_COMMAND_LENGTH = 2
  MAX_COMMAND_LENGTH = 40

  4. Duplicate Code in Rule Matching
  # app/services/commands.py:480-489
  for cand in candidates:
      for i, cmd_text in enumerate(state.texts):
          if cand == cmd_text or cand == state.normalized_texts[i]:
              # ... duplicated code
  - Issue: Same iteration logic repeated 3 times in the file
  - Recommendation: Extract to helper method _find_command_by_text(text, state)

  5. Test Database Pollution
  # test/test_rule_matching.py:46, 58
  service.upload_commands(0, commands)
  - Issue: Test writes to actual database (user_id=0 which is GLOBAL_USER_ID)
  - Recommendation: Use pytest fixtures with transaction rollback or mock database

  ---
  Potential Issues & Risks

  ┌─────────────────────────────┬──────────┬───────────────────────────────────────────────────────────────────────────┐
  │            Risk             │ Severity │                                Description                                │
  ├─────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────┤
  │ Database pollution in tests │ HIGH     │ Tests write to production database (user_id=0)                            │
  ├─────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────┤
  │ Regex performance           │ MEDIUM   │ Large alternation pattern may slow down matching                          │
  ├─────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────┤
  │ Configuration inconsistency │ LOW      │ CLI arg default differs from env var default                              │
  ├─────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────┤
  │ Race condition              │ LOW      │ _prefix_pattern compiled per instance but regex patterns used class-level │
  └─────────────────────────────┴──────────┴───────────────────────────────────────────────────────────────────────────┘

  ---
  Specific Suggestions

  1. Fix Test Database Pollution

  # test/test_rule_matching.py
  import pytest
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  @pytest.fixture
  def test_db():
      engine = create_engine("sqlite:///:memory:")
      SessionLocal = sessionmaker(bind=engine)
      # ... create tables
      yield SessionLocal()
      # cleanup automatically

  2. Extract Helper Method

  # app/services/commands.py
  def _find_command_by_text(self, text: str, state: Bm25MatcherState) -> Optional[CommandMatchResult]:
      for i, cmd_text in enumerate(state.texts):
          if text == cmd_text or text == state.normalized_texts[i]:
              return CommandMatchResult(
                  True, cmd_text, 1.0,
                  command_id=state.command_ids[i],
                  command_code=state.command_codes[i],
              )
      return None

  3. Align Configuration Defaults

  # app/config.py:95
  # Current inconsistency:
  # CLI arg default: uses os.getenv()
  # CommandService.__init__: uses "true" literal

  # Suggested - make consistent:
  default=os.getenv("COMMAND_INTENT_CLASSIFICATION", "true").strip().lower() in {"1", "true", "yes", "on"},

  4. Add Type Hints for New Fields

  # app/services/commands.py:295
  class CommandMatchResult:
      intent_detected: bool = False  # OK
      match_type: Optional[str] = None  # Consider: Optional[Literal["exact", "rule", "fuzzy"]]

  ---
  Test Coverage

  Status: Manual test file added but no pytest integration

  Gaps:
  - No unit tests for _match_rules method
  - No tests for regex edge cases
  - No tests for intent classification boundary conditions

  Recommendations:
  1. Convert test_rule_matching.py to pytest format
  2. Add parametrized tests for regex patterns
  3. Add negative test cases for intent classifier

  ---
  Security Considerations

  No security issues identified in this change.

  ---
  Summary

  ┌───────────────┬────────────────────────────────────────┐
  │   Category    │                 Status                 │
  ├───────────────┼────────────────────────────────────────┤
  │ Functionality │ ✅ Well-implemented two-stage matching │
  ├───────────────┼────────────────────────────────────────┤
  │ Code Style    │ ⚠️ Some duplication, magic numbers     │
  ├───────────────┼────────────────────────────────────────┤
  │ Performance   │ ⚠️ Regex pattern could be optimized    │
  ├───────────────┼────────────────────────────────────────┤
  │ Testing       │ ❌ Tests pollute production database   │
  ├───────────────┼────────────────────────────────────────┤
  │ Security      │ ✅ No concerns                         │
  └───────────────┴────────────────────────────────────────┘

  Recommendation: Address the database pollution in tests before merging, as it could affect production data (user_id=0 commands).