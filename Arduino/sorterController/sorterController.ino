//Small Stepper Motor
#include <Stepper.h>
const int stepsPerRevolution2 = 2038; //steps per revol
Stepper myStepper = Stepper(stepsPerRevolution2, 8, 10, 9, 11);

//gear motor
int motorSpeedPin = 3;
int motorDirection1 = 4;
int motorDirection2 = 5;
String input;


void setup() {
  //use for serial port
  Serial.begin(9600);  
  pinMode(motorSpeedPin, OUTPUT);
  pinMode(motorDirection1, OUTPUT);
  pinMode(motorDirection2, OUTPUT);
}

void loop() {
  
  while(Serial.available()){
    input = Serial.readStringUntil('\n');
    if(input == "e"){
      go(350);
    }else if(input == "o"){
      go(710);
    }else if(input == "r"){
      go(2100);
    }else if(input == "t"){
      go(2100);
    }
  }
}
  void stepper()
  {
  // Rotate CW quickly at 15 RPM
  myStepper.setSpeed(10);
  myStepper.step(-(stepsPerRevolution2 * 0.35));
  delay(1000);

  // Rotate CW quickly at 15 RPM
  myStepper.setSpeed(10);
  myStepper.step((stepsPerRevolution2 * 0.35));
  delay(1000);
  }

  void go(int time)
  {
  analogWrite(motorSpeedPin, 80);
  digitalWrite(motorDirection1, LOW);
  digitalWrite(motorDirection2, HIGH);
  delay(time);
  analogWrite(motorSpeedPin, 0);
  delay(2000);
  stepper();
  delay(2000);

  analogWrite(motorSpeedPin, -80.0);
  digitalWrite(motorDirection1, LOW);
  digitalWrite(motorDirection2, HIGH);
  delay(time/1.5);

  analogWrite(motorSpeedPin, 0);
  delay(3500);
  }
