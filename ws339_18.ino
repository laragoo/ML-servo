double temperature, error;
bool flag = false;
int dt; // period in microseconds set using SET command
int out; // value sent to pin 11

#define DEBUG 1
void userAction() {
  /*
   * This function will be called each time step
   */
  long sum = 0, sumsq = 0;
  int value;
  const int N = 32; /* number of samples measured per time step */
  for (int i = 0; i < N; i++) { /* record N samples */
    value = analogRead(A0);
    sum += value; /* used to calculate mean */
    sumsq += value*(long)value; /* used to calulate variance, need to type cast to long because int is 16 bit */
  }
  double mu, sigma;
  mu = sum / (double) N;
  sigma = sqrt((sumsq - sum*mu)/N); /* expansion of definition of variance */
  const double ADCslope = 4.888e-3; /* 4.888mV / bit */
  const double ADCoffset = 1.02e-3; /* 1.02 mV */
  double vThermo, eVThermo;
  vThermo = ADCslope * mu + ADCoffset; /* convert means in ADC units to voltages */
  eVThermo = ADCslope * sigma;
  const double thermoSlope = 100; /* K/V */
  const double thermoOffset = 273; /* K */
  temperature = thermoSlope * vThermo + thermoOffset;
}

/**********************************************
 * Don't worry about the code below this line *
 * Unless you are curious, but be forewarned  *
 * comments are used very sparingly           *
 **********************************************/

//storage variables
int counter = 0, oc = 0;
int running = 0;
const int N = 16;
unsigned long bytes = 0;
char command[64];
int commandIndex = 0;

void setPeriod (int period) { // period in ms
  const double tick = 1024000 / 16e6; // clock tick in ms
  int number = floor(period / tick) - 1;
  char buf[256];
  sprintf(buf,"DEBUG\tsetPeriod(%d): number = %d, tick = %ld us, actual period = %ld us\n", period, number, (long)(1e3*tick), (long)(1e3*tick*(number+1)));
  Serial.print(buf);
  cli();//stop interrupts
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  OCR1A = number; // = (16*10^6) / (1*1024) - 1 (must be <65536) originally 15624
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // Set CS10 and CS12 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  sei();//allow interrupts
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("WS339-18");
  Serial.flush();
  pinMode(13,OUTPUT);
}

ISR(TIMER1_COMPA_vect){//timer1 interrupt 1Hz toggles pin 13 (LED)
//generates pulse wave of frequency 1Hz/2 = 0.5kHz (takes two cycles for full wave- toggle high then toggle low)
  if (!running) return;
  if (counter % 2){
    digitalWrite(13,HIGH);
  }
  else{
    digitalWrite(13,LOW);
  }
  userAction();
  flag = true;
}

void parseCommand(void) {
  char *ptr;
#ifdef DEBUG
  Serial.print("DEBUG\tparseCommand(): Command = '");
  Serial.println(command);
  Serial.flush();
#endif
  ptr = strtok(command," ");
  if (!ptr) {
#ifdef DEBUG
    Serial.println("DEBUG\tparseCommand(): Verb is NULL");
    Serial.flush();
#endif
    return;
  }
  switch(strlen(ptr)) {
  case 3:
    if (!strcmp(ptr,"SET")) {
      ptr = strtok(NULL," ");
      if (!ptr) {
#ifdef DEBUG
        Serial.println("DEBUG\tparseCommand(): Variable is NULL");
        Serial.flush();
#endif
        return;
      }
#ifdef DEBUG
      Serial.print("DEBUG\tparseCommand(): Variable is ");
      Serial.println(ptr);
      Serial.flush();
#endif
      if (!strcmp("dt",ptr)) {
        ptr = strtok(NULL," ");
        if (running) return;
        if (!ptr) return;
        sscanf(ptr,"%d",&dt);
        return;
      }
      if (!strcmp("out",ptr)) {
        ptr = strtok(NULL," ");
        if (!ptr) return;
        int out;
        sscanf(ptr,"%d",&out);
        analogWrite(11,out);
      }
      }
  case 4:
    if (!strcmp(ptr,"STOP")) {
      running = 0;
      analogWrite(11,0);
      return;
    }
  case 5:
    if (!strcmp(ptr,"START")) {
      setPeriod(dt);
      running = 1;
      return;
    }
  }
#ifdef DEBUG
  Serial.println("DEBUG\tparseCommand(): Unhandled command");
  Serial.flush();
#endif
  return;
}

void getCommand() {
  static unsigned char ch;
  static int discard = 0;
  Serial.readBytes(&ch,1);
#ifdef DEBUG
  Serial.print("DEBUG\tgetCommand() : ch = ");
  Serial.println(ch,BIN);
  Serial.flush();
 #endif
  if (discard && (ch == '\n')) { // reached end of garbage (hopefully \n was not garbage?!
    discard = 0;
    return;
  }
  if (discard) return;
  if (ch & 0x80) {
    command[commandIndex] = 0;
    Serial.print("ERROR\tMSB set in input stream, command so far = \"");
    Serial.print(command);
    Serial.println("\"");
    Serial.flush();
    commandIndex = 0;
    discard = 1;
    return;
  }
  if (ch == '\n') {
    command[commandIndex] = 0;
    parseCommand();
    commandIndex = 0;
    return;
  }
  command[commandIndex++] = ch;
}

void loop() {
  char buf[256];
  if (Serial.available()) getCommand();
  if (flag) {
    Serial.print("T:");
    Serial.println(temperature);
    flag = false;
  }
}
