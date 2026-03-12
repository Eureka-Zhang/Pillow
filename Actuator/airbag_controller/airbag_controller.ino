/*
 * Arduino 气囊控制器
 * 通过串口接收PC端命令控制4个气囊的充气和放气
 * 
 * 引脚对应关系：
 * D2  -- valve1 (放气阀1)    D8  -- pump1 (充气泵1)
 * D3  -- valve2 (放气阀2)    D9  -- pump2 (充气泵2)
 * D4  -- valve3 (放气阀3)    D10 -- pump3 (充气泵3)
 * D5  -- valve4 (放气阀4)    D11 -- pump4 (充气泵4)
 * 
 * 高电平：pump开启(充气)，valve开启(放气)
 * 低电平：pump关闭，valve关闭
 * 
 * 串口命令格式：
 * P<n><t> - 充气命令，n=气囊编号(1-4)，t=档位(1=5秒, 2=10秒)
 * V<n><t> - 放气命令，n=气囊编号(1-4)，t=档位(1=10秒, 2=15秒)
 * S       - 停止所有操作
 * 
 * 示例：
 * P11 - 气囊1充气5秒
 * P12 - 气囊1充气10秒
 * V21 - 气囊2放气10秒
 * V22 - 气囊2放气15秒
 */

// 引脚定义
const int PUMP_PINS[4] = {8, 9, 10, 11};  // pump1, pump2, pump3, pump4
const int VALVE_PINS[4] = {2, 3, 4, 5};   // valve1, valve2, valve3, valve4

// 时间配置（毫秒）
const unsigned long INFLATE_TIMES[2] = {5000, 10000};   // 充气：5秒, 10秒
const unsigned long DEFLATE_TIMES[2] = {10000, 15000};  // 放气：10秒, 15秒

// 每个气囊独立状态：可多气囊同时充气/放气；同一气囊重复命令仅重置该气囊计时（非阻塞）
enum AirbagAction { ACTION_IDLE = 0, ACTION_INFLATE = 1, ACTION_DEFLATE = 2 };

AirbagAction airbagAction[4] = { ACTION_IDLE, ACTION_IDLE, ACTION_IDLE, ACTION_IDLE };
unsigned long airbagEndTime[4] = { 0, 0, 0, 0 };

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 4; i++) {
    pinMode(PUMP_PINS[i], OUTPUT);
    pinMode(VALVE_PINS[i], OUTPUT);
    digitalWrite(PUMP_PINS[i], LOW);
    digitalWrite(VALVE_PINS[i], LOW);
  }
  Serial.println("Airbag Controller Ready");
  Serial.println("Commands: P<n><t> (inflate), V<n><t> (deflate), S (stop)");
  Serial.println("Per-airbag timing; same airbag resets timer only.");
}

void loop() {
  unsigned long now = millis();
  for (int i = 0; i < 4; i++) {
    if (airbagAction[i] != ACTION_IDLE && now >= airbagEndTime[i]) {
      stopAirbag(i + 1);
    }
  }

  // 至少需要 3 字节才尝试解析（P/V/S + 两位数字）
  if (Serial.available() < 3) return;

  // 首字节必须是 P/V/S；否则只丢弃这 1 字节并 return，避免一次吞掉多字节导致整条命令丢失（如 "11p" 被读光后无法解析）
  char cmd = Serial.read();
  if (cmd != 'P' && cmd != 'p' && cmd != 'V' && cmd != 'v' && cmd != 'S' && cmd != 's') return;

  // 取第二个逻辑字符：气囊号 1-4，跳过空格/换行
  char airbagChar = 0;
  while (Serial.available() > 0) {
    airbagChar = Serial.read();
    if (airbagChar >= '1' && airbagChar <= '4') break;
  }
  if (airbagChar < '1' || airbagChar > '4') {
    if (cmd == 'S' || cmd == 's') {
      Serial.println("RX:S");
      stopAllOperations();
      Serial.println("OK:All operations stopped");
      while (Serial.available() > 0) {
        char c = Serial.peek();
        if (c != ' ' && c != '\r' && c != '\n') break;
        Serial.read();
      }
    }
    return;
  }

  // S 命令：不需要两位数字，有 S 就停
  if (cmd == 'S' || cmd == 's') {
    Serial.println("RX:S");
    stopAllOperations();
    Serial.println("OK:All operations stopped");
    while (Serial.available() > 0) {
      char c = Serial.peek();
      if (c != ' ' && c != '\r' && c != '\n') break;
      Serial.read();
    }
    return;
  }

  // 取第三个逻辑字符：档位 1-2，跳过空格/换行
  char durationChar = 0;
  while (Serial.available() > 0) {
    durationChar = Serial.read();
    if (durationChar == '1' || durationChar == '2') break;
  }
  // 只丢弃空白，避免把下一条命令的开头（如 p11）一起清掉导致后续再也凑不齐 3 字节
  while (Serial.available() > 0) {
    char c = Serial.peek();
    if (c != ' ' && c != '\r' && c != '\n') break;
    Serial.read();
  }

  int airbag = airbagChar - '0';
  int level = durationChar == '2' ? 2 : 1;  // 未收到有效档位时视为 1
  if (level != 1 && level != 2) {
    Serial.println("ERR:Invalid level (1 or 2)");
    return;
  }

  // 回显收到的命令
  Serial.print("RX:");
  Serial.print((char)cmd);
  Serial.print(airbag);
  Serial.println(level);
  if (cmd == 'P' || cmd == 'p') {
    startInflate(airbag, level);
  } else {
    startDeflate(airbag, level);
  }
}

void startInflate(int airbag, int level) {
  int idx = airbag - 1;
  unsigned long duration = INFLATE_TIMES[level - 1];
  digitalWrite(VALVE_PINS[idx], LOW);
  digitalWrite(PUMP_PINS[idx], HIGH);
  airbagAction[idx] = ACTION_INFLATE;
  airbagEndTime[idx] = millis() + duration;
  Serial.print("OK:Inflating airbag ");
  Serial.print(airbag);
  Serial.print(" for ");
  Serial.print(duration / 1000);
  Serial.println(" seconds");
}

void startDeflate(int airbag, int level) {
  int idx = airbag - 1;
  unsigned long duration = DEFLATE_TIMES[level - 1];
  digitalWrite(PUMP_PINS[idx], LOW);
  digitalWrite(VALVE_PINS[idx], HIGH);
  airbagAction[idx] = ACTION_DEFLATE;
  airbagEndTime[idx] = millis() + duration;
  Serial.print("OK:Deflating airbag ");
  Serial.print(airbag);
  Serial.print(" for ");
  Serial.print(duration / 1000);
  Serial.println(" seconds");
}

void stopAirbag(int airbag) {
  int idx = airbag - 1;
  if (idx < 0 || idx > 3) return;
  if (airbagAction[idx] == ACTION_INFLATE) {
    Serial.print("OK:Airbag ");
    Serial.print(airbag);
    Serial.println(" finished inflating");
  } else if (airbagAction[idx] == ACTION_DEFLATE) {
    Serial.print("OK:Airbag ");
    Serial.print(airbag);
    Serial.println(" finished deflating");
  }
  digitalWrite(PUMP_PINS[idx], LOW);
  digitalWrite(VALVE_PINS[idx], LOW);
  airbagAction[idx] = ACTION_IDLE;
  airbagEndTime[idx] = 0;
}

void stopAllOperations() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(PUMP_PINS[i], LOW);
    digitalWrite(VALVE_PINS[i], LOW);
    airbagAction[i] = ACTION_IDLE;
    airbagEndTime[i] = 0;
  }
}
