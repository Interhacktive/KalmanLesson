#include "testApp.h"

using namespace ofxCv;
using namespace cv;

// based on code from:
// http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/

void testApp::setup() {
	ofSetVerticalSync(true);
    ofEnableSmoothing();
    
	
	KF.init(4, 2, 0);
	
	KF.transitionMatrix = *(Mat_<float>(4, 4) <<
                            1,0,1,0,
                            0,1,0,1,
                            0,0,1,0,
                            0,0,0,1);
    
	measurement = Mat_<float>::zeros(2,1);
    
	KF.statePre.at<float>(0) = mouseX;
	KF.statePre.at<float>(1) = mouseY;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
    
    
    
    
    //UI
	ofEnableSmoothing(); 
    
    //set some sketch parameters
    /*
    //Background Color 
    red = 233; 
    blue = 240; 
    green = 52; 
    alpha = 200; 
    radius = 150; 
    drawFill = true;     
    backgroundColor = ofColor(233, 52, 27); 
    resolution = 30; 
    position = ofPoint(ofGetWidth()*.5, ofGetHeight()*.5); 
     */
    ofSetCircleResolution(10); 
    float dim = 16; 
	float xInit = OFX_UI_GLOBAL_WIDGET_SPACING; 
    float length = 320-xInit; 
	
    gui = new ofxUICanvas(0,0,length+xInit*2.0,ofGetHeight());
    gui->addLabel("Kalman Filter Adjustments", OFX_UI_FONT_LARGE);    
    gui->addSpacer(length, 2);
    gui->addSpacer(length, 2);
    gui->addSlider("processNoiseCovMin", 0, 10000, processNoiseCovMin, length, dim);
    gui->addSlider("processNoiseCovMax", 0, 10000, processNoiseCovMax, length, dim);
    gui->addSpacer(length, 2);
    gui->addSlider("measurementNoiseCovMin", 0, 10000, measurementNoiseCovMin, length, dim);
    gui->addSlider("measurementNoiseCovMax", 0, 10000, measurementNoiseCovMax, length, dim);
    gui->addSpacer(length, 2);
    gui->addSlider("errorCovPost", 0, 5, errorCovPost, length, dim);
    gui->addSpacer(length, 2);
    gui->addLabel("Cursor Adjustments");
    gui->addSpacer(length, 2); 
    gui->addSpacer(length, 2);
    gui->addSlider("randomAmount", 0, 100, randomAmount, length, dim);
    gui->addSlider("cursorInput", 0, 1000, cursorInput, length, dim);
    gui->addSpacer(length, 2);
    gui->addToggle( "Show Raw", showRaw, dim, dim);
    gui->addToggle( "Show Predicted", showPredicted, dim, dim);
    gui->addToggle( "Show Smoothed", showSmoothed, dim, dim);
    gui->addSlider("Curve Smooth Amount", 0, 50, curveSmoothAmount, length, dim);
    gui->addSpacer(length, 2);
    gui->addWidgetDown(new ofxUIToggle(32, 32, false, "FULLSCREEN"));
    
   // gui->addWidgetDown(new ofxUIMovingGraph(length-xInit, 120, cursorDistance, 256, 0, 400, "MOVING GRAPH"));
     gui->addLabel("Cursor Distance", OFX_UI_FONT_MEDIUM);
    vector<float> buffer; 
    for(int i = 0; i < 256; i++)
    {
        buffer.push_back(0.0);
    }
      mg = (ofxUIMovingGraph *) gui->addWidgetDown(new ofxUIMovingGraph(length-xInit, 120, buffer, 256, 0, 400, "MOVING GRAPH"));
    
    ofAddListener(gui->newGUIEvent,this,&testApp::guiEvent);	
	ofBackground(backgroundColor); 
    
    gui->loadSettings("GUI/guiSettings.xml");
    gui->setTheme(3);

    
}

void testApp::update() {
    
    setIdentity(KF.measurementMatrix);
    
	setIdentity(KF.processNoiseCov, Scalar::all(ofMap(cursorDistance, 0, cursorInput, processNoiseCovMax, processNoiseCovMin)));//1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(ofMap(cursorDistance, 0, cursorInput, measurementNoiseCovMax, measurementNoiseCovMin)));//1e-1));10000
    setIdentity(KF.errorCovPost, Scalar::all(errorCovPost));//.1));
    
	// First predict, to update the internal statePre variable
	Mat prediction = KF.predict();
	cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
	point = toOf(predictPt);
	predicted.addVertex(point);
	
	// Get mouse point
	measurement(0) = mouseX+ofRandom(randomAmount);
	measurement(1) = mouseY+ofRandom(randomAmount);
	
	cv::Point measPt(measurement(0),measurement(1));
	line.addVertex(toOf(measPt));
	
	// The "correct" phase that is going to use the predicted value and our measurement
	Mat estimated = KF.correct(measurement);
	cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    ofPoint newPoint = toOf(measPt);
    cursorDistance = ofDist(point.x, point.y, mouseX, mouseY);
    mg->addPoint(cursorDistance);
    
    predicted.addVertex(point);
    
    smoothLine = predicted.getSmoothed(curveSmoothAmount);
    
}

void testApp::draw() {
	ofBackground(0);

    
    ofPushStyle();
    ofNoFill();
    
    if(showRaw){       
        //draw raw input
        ofSetColor(ofColor::blue);
        line.draw();
        ofRect(mouseX,mouseY, randomAmount, randomAmount);
    
    }
    if(showPredicted){
        //draw predicted output
        ofSetColor(ofColor::red);
        predicted.draw();
        ofCircle(point, randomAmount/2);
    }
    
    if(showSmoothed){
        ofSetColor(ofColor::green);
        smoothLine.draw();
    }
        
  
    ofPopStyle();

}

//--------------------------------------------------------------
void testApp::guiEvent(ofxUIEventArgs &e)
{
	string name = e.widget->getName(); 
	int kind = e.widget->getKind(); 
	
	if(name == "processNoiseCovMin")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		processNoiseCovMin= slider->getScaledValue(); 
	}
    if(name == "processNoiseCovMax")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		processNoiseCovMax = slider->getScaledValue(); 
	}
	else if(name == "measurementNoiseCovMin")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		measurementNoiseCovMin = slider->getScaledValue(); 
	}	
    else if(name == "measurementNoiseCovMax")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		measurementNoiseCovMax = slider->getScaledValue(); 
	}	
	else if(name == "errorCovPost")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		errorCovPost = slider->getScaledValue(); 		
	}
    else if(name == "randomAmount")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		randomAmount = slider->getScaledValue(); 		
	}

    else if(name == "cursorInput")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		cursorInput = slider->getScaledValue(); 		
	}
    else if(name == "Show Raw")
	{
		ofxUIToggle *toggle = (ofxUIToggle *) e.widget; 
		showRaw = toggle->getValue(); 
	}
    else if(name == "Show Predicted")
	{
		ofxUIToggle *toggle = (ofxUIToggle *) e.widget; 
		showPredicted = toggle->getValue(); 
	}
    else if(name == "Show Smoothed")
	{
		ofxUIToggle *toggle = (ofxUIToggle *) e.widget; 
		showSmoothed = toggle->getValue(); 
	}
    else if(e.widget->getName() == "FULLSCREEN")
    {
        ofxUIToggle *toggle = (ofxUIToggle *) e.widget;
        ofSetFullscreen(toggle->getValue());   
    }
    else if(name == "Curve Smooth Amount")
	{
		ofxUISlider *slider = (ofxUISlider *) e.widget; 
		curveSmoothAmount = slider->getScaledValue(); 		
	}
        
}

void testApp::exit(){
     gui->saveSettings("GUI/guiSettings.xml");
    delete gui; 
    //delete[] buffer; 
}
//--------------------------------------------------------------


//--------------------------------------------------------------
void testApp::keyPressed(int key)
{
    switch (key) 
    {            
        case 'h':
        {
            gui->toggleVisible(); 
        }
            break; 
        case 'f':
			ofToggleFullscreen(); 
			break;
        default:
            break;
    }
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    predicted.clear();
    line.clear();

    
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}