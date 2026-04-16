# 语音指令识别优化实施计划 (acc-opt)

## 1. 目标 (Objective)
通过基于规则的意图识别和指令匹配，提高 OrangePi 设备在复杂环境下的指令识别精度，同时支持普通语音转录。

## 2. 关键文件与上下文 (Key Files & Context)
*   `app/services/commands.py`: 现有的匹配逻辑实现。
*   `docs/zhiling.txt`: 目标识别指令列表。
*   `config/command_text_aliases.json`: 现有的文本别名和替换规则。
*   `app/api/ws.py` & `app/services/voice/session.py`: ASR 会话流，集成指令识别的入口。

## 3. 实施步骤 (Implementation Steps)

### 3.1 预处理与模式提取 (Preprocessing)
1.  分析 `docs/zhiling.txt`，提取核心任务、阶段和前缀。
2.  将提取的模式转化为正则表达式和关键词列表。
3.  在 `config/command_text_aliases.json` 中补充 `zhiling.txt` 中发现的常见错误同音词（如“个号”->“各号”，“庭”->“停”）。

### 3.2 意图识别器开发 (Intent Classifier)
1.  在 `app/services/commands.py` 中新增 `IntentClassifier` 类。
2.  实现基于长度、关键词（如：检查、准备、停、起飞等）和正则表达式的意图判定逻辑。
3.  增加配置项 `ENABLE_INTENT_CLASSIFICATION` 以便灵活切换。

### 3.3 指令匹配引擎优化 (Command Matcher)
1.  **第一阶段 (Exact/Alias Match)**: 
    *   先进行规范化（利用 `command_text_aliases.json`）。
    *   尝试直接匹配指令库。
2.  **第二阶段 (Template Match)**:
    *   使用正则表达式匹配 `[前缀]?[任务][阶段]?` 结构的文本。
    *   处理缺省情况（如：只有“五分钟准备”，根据上下文或任务列表尝试匹配）。
3.  **第三阶段 (Fuzzy Match Fallback)**:
    *   仅在意图判定为“指令”且规则未完全匹配时，触发 BM25+RapidFuzz 匹配。
    *   调高模糊匹配的阈值（如从 0.75 提升至 0.85）以减少误报。

### 3.4 集成与配置 (Integration)
1.  更新 `CommandService.match_command` 方法，集成意图识别和多阶段匹配。
2.  在 `app/config.py` 中暴露相关参数（意图识别开关、匹配模式路径等）。

### 3.5 验证与测试 (Verification)
1.  编写单元测试，使用 `zhiling.txt` 中的样本进行验证。
2.  编写负样本测试（非指令语句），验证是否被正确判断为“非指令”并仅进行转录。
3.  在 OrangePi 上进行端到端测试，观察实时响应和 CPU 占用。

## 4. 验证与测试 (Verification & Testing)
*   **指令覆盖率**: 50条核心指令在理想环境下识别率 > 99%。
*   **误报率**: 日常对话被错误识别为指令的概率 < 2%。
*   **性能**: 意图识别和匹配过程的延迟增加 < 5ms。
