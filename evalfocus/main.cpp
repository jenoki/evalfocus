//
//  main.cpp
//  evalfocus
//
//  Created by Junya Enoki on 2018/11/18.
//  Copyright © 2018 ruffles inc. All rights reserved.
//

#include <iostream>
#include <unistd.h>
#include <thread>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;
const string default_front_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
const string default_profile_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_profileface.xml";
const string default_eyes_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
const double process_version = 0.1;

extern char *optarg;
extern int optind, opterr;
#if 0
void frontal_face_detect(CascadeClassifier front_face_cascade, Mat source_image, vector<Rect> front_faces, double scale_factor, int min_neighbors, int face_flag, Size min_face){
    front_face_cascade.detectMultiScale(source_image, front_faces, scale_factor, min_neighbors, face_flag , min_face);
}

void profile_face_detect(CascadeClassifier profile_face_cascade,Mat source_image, vector<Rect> profile_faces, double scale_factor, int min_neighbors, int face_flag, Size min_face){
    profile_face_cascade.detectMultiScale(source_image, profile_faces, scale_factor, min_neighbors, face_flag , min_face);
}
#endif
int main(int argc, char * argv[]) {
    CascadeClassifier front_face_cascade;
    CascadeClassifier profile_face_cascade;
    CascadeClassifier eyes_cascade;

    Mat source_image;
    string source_file;
    string front_face_cascade_file;
    string profile_face_cascade_file;
    string eyes_cascade_file;
    string log_file;
    ofstream clog;
    
    // Command Line Analyze.
    int op;
    opterr = 0;
    if (argc == 1){
        cerr << "Usage:evalfocus [-c cascade] [-l logfile] [-f] <image>" << endl;
        exit(0);
    }
    
    while ( (op = getopt(argc,argv,"f:c:l:")) != -1){
        switch (op){
            case 'f':
                source_file = optarg;
                break;
            case 'c':
                front_face_cascade_file = optarg;
                break;
            case 'l':
                log_file = optarg;
                break;
        }
    }
    
    if (source_file.empty()){
        source_file = argv[1];
    }

    //Load cascades for faces.
    if (front_face_cascade_file.empty()){
        front_face_cascade_file = default_front_cascade_file;
    }
    if (!front_face_cascade.load(front_face_cascade_file)){
        cerr << "front face cascade file not found." << endl;
        exit(1);
    }

    //Load cascades for profile faces.
    if (profile_face_cascade_file.empty()){
        profile_face_cascade_file = default_profile_cascade_file;
    }
    if (!profile_face_cascade.load(profile_face_cascade_file)){
        cerr << "profile face cascade file not found." << endl;
        exit(1);
    }

    //Load cascades for eyes.
    eyes_cascade_file = default_eyes_cascade_file;
    if (!eyes_cascade.load(eyes_cascade_file)){
        cerr << "eyes cascade file not found." << endl;
        exit(1);
    }
    //Create log file
    if (!log_file.empty()){
        clog.open(log_file);
        if (clog.is_open()) clog << "Process_Version:" << process_version << endl;
    }

    //Load source image
    source_image = imread(source_file,IMREAD_GRAYSCALE);
    if (source_image.empty()) {
        cerr << "image read failed." << endl;
        exit(1);
    }
    
    //Detect faces
    const int min_neighbors = 3;
    const double scale_factor = 1.16;
    int longside = max(source_image.size().width,source_image.size().height);
    int rectsize = longside / 32; //minimum size of face detect
    int face_flag = CASCADE_FIND_BIGGEST_OBJECT;
    vector<Rect> front_faces;
    vector<Rect> profile_faces;

    Size min_face = Size(rectsize,rectsize);
    //Front face
    front_face_cascade.detectMultiScale(source_image, front_faces, scale_factor, min_neighbors, face_flag , min_face);
    //Profile face
    profile_face_cascade.detectMultiScale(source_image, profile_faces, scale_factor, min_neighbors, face_flag , min_face);

    cout << "front_faces:"<< front_faces.size() << endl;
    if (clog.is_open()) clog << "front_faces:" << front_faces.size() << endl;

    cout << "profile_faces:"<< profile_faces.size() << endl;
    if (clog.is_open()) clog << "profile_faces:" << profile_faces.size() << endl;

    if ( (front_faces.size() == 0) && (profile_faces.size() == 0) ) {
        cerr << "faces not found." << endl;
        exit(0);
    }
    
    //Marge front_faces & profile faces
    front_faces.insert(front_faces.end(),profile_faces.begin(),profile_faces.end());

    //Loop for found faces
    double max_avg = 0.0; //Max average value from faces
    vector<Rect> eyes;
    
    for (int i = 0; i < front_faces.size(); i++){
        //Parameter for eye detection
        int eye_factor = 16;
        int eye_flag = CASCADE_FIND_BIGGEST_OBJECT;

        //Detect eyes from current face
        Point origin(front_faces[i].x,front_faces[i].y);
        Size area(front_faces[i].width,front_faces[i].height);
        Mat roi_face = source_image(Rect(origin, area)); //For detect eyes.
        Size min_eyes = Size(front_faces[i].width/eye_factor, front_faces[i].height/eye_factor);
        eyes_cascade.detectMultiScale(roi_face, eyes, scale_factor, min_neighbors, eye_flag, min_eyes);

        cout << "face:" << (i+1) << " eyes:" << eyes.size();
        if (clog.is_open()) clog << "face:" << (i+1) << " eyes:" << eyes.size();

        //If found eye, evaluate focus accuracy from face.
        int optx = getOptimalDFTSize(roi_face.rows);
        int opty = getOptimalDFTSize(roi_face.cols);
        Mat padded_face;
        copyMakeBorder(roi_face, padded_face, 0, optx - roi_face.rows, 0, opty - roi_face.cols, BORDER_CONSTANT, Scalar::all(0));
        
        Mat planes[] = {Mat_<float>(padded_face), Mat::zeros(padded_face.size(), CV_32F)};
        Mat complexI;
        merge(planes, 2, complexI);  // Add to the expanded another plane with zeros
        dft(complexI, complexI);     // this way the result may fit in the source matrix
        split(complexI, planes);     // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        Mat magI = planes[0];

        magI += Scalar::all(1);      // switch to logarithmic scale
        log(magI, magI);
        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

        normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
        // viewable image form (float between values 0 and 1).

        //Get high frequency area from DFT result.
        //Make mask image
        Mat mask = Mat::zeros(Size(magI.cols,magI.rows), CV_8U);
        Point center = Point(magI.cols/2, magI.rows/2);
        int r = (magI.cols/2) * 0.9;

        circle(mask, center, r ,Scalar(255,255,255,0), -1, LINE_8);
        double hf_avg = mean(magI, mask)[0];

        if (max_avg < hf_avg) max_avg = hf_avg;

        cout << " avg:" << hf_avg << endl;
        if (clog.is_open()) clog << "avg:" << hf_avg << endl;


    }
    //Output result
    int accuracy = (int)((max_avg * 100.0) + 0.5);
    cout << "accuracy:" << accuracy << endl;
    if (clog.is_open()) clog << "accuracy:" << accuracy << endl;
    exit(accuracy);
}
