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

// 操作状态
unsigned long operationEndTime = 0;
int activePump = -1;
int activeValve = -1;
bool operationInProgress = false;

void setup() {
  // 初始化串口
  Serial.begin(9600);
  
  // 初始化所有引脚为输出，并设置为低电平（关闭状态）
  for (int i = 0; i < 4; i++) {
    pinMode(PUMP_PINS[i], OUTPUT);
    pinMode(VALVE_PINS[i], OUTPUT);
    digitalWrite(PUMP_PINS[i], LOW);
    digitalWrite(VALVE_PINS[i], LOW);
  }
  
  Serial.println("Airbag Controller Ready");
  Serial.println("Commands: P<n><t> (inflate), V<n><t> (deflate), S (stop)");
  Serial.println("n=1-4 (airbag), t=1(5s/10s) or 2(10s/15s)");
}

void loop() {
  // 检查是否有操作需要结束
  if (operationInProgress && millis() >= operationEndTime) {
    stopAllOperations();
    Serial.println("OK:Operation completed");
  }
  
  // 处理串口命令
  if (Serial.available() >= 3) {
    char cmd = Serial.read();
    char airbagChar = Serial.read();
    char durationChar = Serial.read();
    
    // 清除串口缓冲区中的多余字符（如换行符）
    while (Serial.available() > 0) {
      Serial.read();
    }
    
    int airbag = airbagChar - '0';
    int level = durationChar - '0';
    
    // 验证参数
    if (airbag < 1 || airbag > 4) {
      Serial.println("ERR:Invalid airbag number (1-4)");
      return;
    }
    
    if (level != 1 && level != 2) {
      Serial.println("ERR:Invalid level (1 or 2)");
      return;
    }
    
    // 执行命令
    if (cmd == 'P' || cmd == 'p') {
      // 充气命令
      startInflate(airbag, level);
    } else if (cmd == 'V' || cmd == 'v') {
      // 放气命令
      startDeflate(airbag, level);
    } else if (cmd == 'S' || cmd == 's') {
      // 停止命令
      stopAllOperations();
      Serial.println("OK:All operations stopped");
    } else {
      Serial.println("ERR:Unknown command");
    }
  }
}

void startInflate(int airbag, int level) {
  // 先停止所有操作
  stopAllOperations();
  
  int pumpIndex = airbag - 1;
  int timeIndex = level - 1;
  unsigned long duration = INFLATE_TIMES[timeIndex];
  
  // 开启对应的充气泵
  digitalWrite(PUMP_PINS[pumpIndex], HIGH);
  activePump = pumpIndex;
  
  // 设置结束时间
  operationEndTime = millis() + duration;
  operationInProgress = true;
  
  Serial.print("OK:Inflating airbag ");
  Serial.print(airbag);
  Serial.print(" for ");
  Serial.print(duration / 1000);
  Serial.println(" seconds");
}

void startDeflate(int airbag, int level) {
  // 先停止所有操作
  stopAllOperations();
  
  int valveIndex = airbag - 1;
  int timeIndex = level - 1;
  unsigned long duration = DEFLATE_TIMES[timeIndex];
  
  // 开启对应的放气阀
  digitalWrite(VALVE_PINS[valveIndex], HIGH);
  activeValve = valveIndex;
  
  // 设置结束时间
  operationEndTime = millis() + duration;
  operationInProgress = true;
  
  Serial.print("OK:Deflating airbag ");
  Serial.print(airbag);
  Serial.print(" for ");
  Serial.print(duration / 1000);
  Serial.println(" seconds");
}

void stopAllOperations() {
  // 关闭所有泵和阀
  for (int i = 0; i < 4; i++) {
    digitalWrite(PUMP_PINS[i], LOW);
    digitalWrite(VALVE_PINS[i], LOW);
  }
  
  activePump = -1;
  activeValve = -1;
  operationInProgress = false;
}
