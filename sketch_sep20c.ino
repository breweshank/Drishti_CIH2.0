int eegPin = A5;  // Assign the analog pin
void setup() {
    Serial.begin(9600);  // Initialize serial communication
}

void loop() {
    int eegValue = analogRead(eegPin);  // Read the EEG signal
    Serial.println(eegValue);  // Send the value to the serial monitor
    delay(1);  // Adjust as necessary
}
