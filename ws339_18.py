# -*- coding: utf-8 -*-

import serial
import matplotlib.pyplot as p
import numpy

class Arduino:
    def __init__(self,verbose=0):
        self.verbose = verbose
        if verbose: print "introArduino class creator: Verbose mode activated"
        for i in range(10):
            device = "/dev/ttyACM%d" % (i)
            try:
                self.device = serial.Serial(device,baudrate=115200,timeout=1.0)
                if verbose: print "Found device at %s" % (device)
                break
            except:
                continue
        self.device.setDTR(1); #reboot Arduino
        self.device.setDTR(0);
        self.device.setDTR(1);
        exception_count = 0
        attempts = 0
        while True:
            try:
                if "WS339-18" == self.getResp()[0:8]:
                    if verbose: print "Arduino is communicating"
                    return
            except:
                if self.verbose: print "Exception"
                exception_count = exception_count + 1
            attempts = attempts + 1
            if 5 == attempts:
                print "Unable to communicate with Arduino...%d exceptions" % (exception_count)
                exit
    def send(self,str):
        self.device.write("%s\n" % (str))
        if self.verbose: print "Sent '%s'" % (str)
    def getResp(self):
        if self.verbose: print "Waiting for response..."
        str = self.device.readline()
        str = str.replace("\n","")
        str = str.replace("\r","")
        if self.verbose: print "Got response: '%s'" % (str)
        return str
