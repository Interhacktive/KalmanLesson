#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxUI.h"

class testApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
        void exit();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
    //Kalman Filter
	cv::KalmanFilter KF;
	cv::Mat_<float> measurement;
	
	ofPolyline predicted, line, smoothLine;
    
	ofVec2f point, pPoint, ppPoint;
    
    int pMouseX, pMouseY;
    float inMax, outMax;
    
    //UI
    ofxUIMovingGraph *mg; 
    ofxUICanvas *gui;   	
	void guiEvent(ofxUIEventArgs &e);
    bool drawFill; 
	float red, green, blue, alpha; 	
    
    ofColor backgroundColor; 
    float radius; 
    int resolution;    
    ofPoint position; 
    
    float processNoiseCovMin, processNoiseCovMax;
    float measurementNoiseCovMin, measurementNoiseCovMax;
    float errorCovPost;
    float randomAmount;
    float cursorDistance;
    float cursorInput;
    bool showRaw, showPredicted, showSmoothed;
    int curveSmoothAmount;
     float *buffer;
    
};
