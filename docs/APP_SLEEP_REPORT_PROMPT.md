# Prompt：App 端接收睡眠报告数据功能

按以下规范实现「接收并展示睡眠报告」功能。

---

## 一、连接与协议概要

- **连接方式**：App 通过 **TCP** 连接枕头端总控（默认端口 **8765**），或通过蓝牙 RFCOMM（协议一致）。
- **协议**：**按行** 收发，每行一条消息，行尾为 `\n`（UTF-8）。
- **流程**：App 发送一行指令 → 枕头端返回 **一行或多行** 回复，每行一条。

---

## 二、与睡眠报告相关的指令与回复

### 2.1 结束睡眠并获取报告

- **App 发送**（一行）：  
  `STOP_SLEEP`
- **枕头端返回**（两行，按顺序）：
  1. `OK:STOP_SLEEP`
  2. `SLEEP_REPORT:` + **一整行 JSON 字符串**（无换行，同一行内）

因此 App 在发送 `STOP_SLEEP` 后，需要**按行读取**，先读到 `OK:STOP_SLEEP`，再读下一行；若该行以 `SLEEP_REPORT:` 开头，则去掉此前缀后，对剩余部分做 **JSON 解析**，得到睡眠报告对象。

---

## 三、SLEEP_REPORT JSON 字段说明

解析 `SLEEP_REPORT:` 后的 JSON 可得到如下结构（均为服务端实际返回字段，建议 App 按此接收并展示）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `started_at` | string | 睡眠开始时间，格式 `"YYYY-MM-DD HH:MM:SS"` |
| `ended_at` | string | 睡眠结束时间，格式 `"YYYY-MM-DD HH:MM:SS"` |
| `duration_sec` | int | 睡眠时长（秒），由开始/结束时间计算 |
| `total_sleep_sec` | float | 睡眠总时长（秒），与枕头实际监测时长一致 |
| `snoring_total_sec` | float | 打鼾总时长（秒） |
| `snoring_posture_ratio` | object | 打鼾时的睡姿占比，键为睡姿名称，值为 0~1 的小数，总和为 1 |
| `sleep_score` | int | 睡眠评分，0~100，越高越好（0% 打鼾≈100 分，100% 打鼾≈40 分） |
| `snore_events` | int | 打鼾事件次数（触发止鼾逻辑的次数） |
| `auto_adjust_events` | int | 自动调节次数 |
| `manual_adjust_events` | int | 手动调节次数 |
| `reset_count` | int | 一键复位次数 |
| `command_count` | int | 总指令次数 |
| `alarms` | array of string | 闹钟列表，如 `["07:00"]` |

### 3.1 `snoring_posture_ratio` 结构示例

睡姿名称为英文，与枕头端一致，例如：

```json
{
  "Middle": 0.45,
  "Left-Back": 0.2,
  "Right-Back": 0.15,
  "Left": 0.1,
  "Right": 0.1
}
```

App 可据此展示「打鼾时各睡姿占比」（如饼图或条形图），并可自行做中文映射（如 Middle → 仰卧正中、Left → 左侧卧 等）。

---

## 四、App 端需要实现的功能要点

1. **发送 `STOP_SLEEP` 后**  
   - 按行读取回复；  
   - 识别第一行为 `OK:STOP_SLEEP`；  
   - 将下一行识别为睡眠报告：若以 `SLEEP_REPORT:` 开头，则去掉该前缀后对整行做 JSON 解析。

2. **解析后的展示建议**  
   - **睡眠总时长**：用 `total_sleep_sec` 或 `duration_sec` 换算为「时:分」或「X 小时 Y 分钟」展示。  
   - **打鼾时长**：用 `snoring_total_sec` 换算并展示，可与总时长一起展示「打鼾占比」。  
   - **打鼾时睡姿占比**：用 `snoring_posture_ratio` 做饼图/条形图等，键可做本地化。  
   - **睡眠评分**：用 `sleep_score`（0–100）展示，可配合简单等级（如 90–100 优秀、70–89 良好等）。

3. **健壮性**  
   - 若某次返回缺少 `total_sleep_sec`、`snoring_total_sec`、`snoring_posture_ratio`、`sleep_score`，可默认为 0 或空对象，避免崩溃。  
   - 若返回的不是两行或第二行不是 `SLEEP_REPORT:` 格式，应做错误提示或重试逻辑（由 App 自行决定）。

---

## 五、完整 SLEEP_REPORT 示例（供联调参考）

```
OK:STOP_SLEEP
SLEEP_REPORT:{"started_at": "2026-03-12 08:00:00", "ended_at": "2026-03-12 09:30:00", "duration_sec": 5400, "snore_events": 3, "auto_adjust_events": 2, "manual_adjust_events": 0, "reset_count": 0, "command_count": 5, "alarms": ["07:00"], "total_sleep_sec": 5398.5, "snoring_total_sec": 120.5, "snoring_posture_ratio": {"Middle": 0.45, "Left-Back": 0.2, "Right-Back": 0.15, "Left": 0.1, "Right": 0.1}, "sleep_score": 99}
```

App 只需实现：**TCP/蓝牙连接 → 按行收发 → 发送 STOP_SLEEP → 解析两行回复中的 SLEEP_REPORT JSON → 用上述字段做界面展示** 即可完成「接收数据」功能。
