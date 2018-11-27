//
//  main.cpp
//  evalfocus
//
//  Created by Junya Enoki on 2018/11/18.
//  Copyright © 2018 ruffles inc. All rights reserved.
//

#include <iostream>
#include <unistd.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
string default_face_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
string default_eyes_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_eye.xml";
const double process_version = 0.1;

extern char *optarg;
extern int optind, opterr;

int main(int argc, char * argv[]) {
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    vector<Rect> faces;
    vector<Rect> eyes;
    Size min_face;
    Size min_eyes;
    const int min_neighbors = 3;
    const double scale_factor = 1.15;

    Mat source_image;
    string source_file;
    string face_cascade_file;
    string eyes_cascade_file;
    string log_file;
    ofstream clog;
    
    int op;
    opterr = 0;
    
    // Command Line Analyze.
    
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
                face_cascade_file = optarg;
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
    if (face_cascade_file.empty()){
        face_cascade_file = default_face_cascade_file;
    }
    if (!face_cascade.load(face_cascade_file)){
        cerr << "face cascade file not found." << endl;
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
    
    //Detect faces
    int longside = max(source_image.size().width,source_image.size().height);
    int rectsize = longside / 32; //minimum size of face detect
    int face_flag = CASCADE_FIND_BIGGEST_OBJECT;
    min_face = Size(rectsize,rectsize);

    face_cascade.detectMultiScale(source_image, faces, scale_factor, min_neighbors, face_flag , min_face);
    cout << "detected_faces:"<< faces.size() << endl;
    if (clog.is_open()) clog << "detected_faces:"<< faces.size() << endl;

    if (faces.size() == 0) {
        exit(0);
    }
    
    //Loop for found faces
    double max_avg = 0.0; //Max average value from faces
    for (int i = 0; i < faces.size(); i++){
        //Detect eyes from current face
        Point origin(faces[i].x,faces[i].y);
        Size area(faces[i].width,faces[i].height);
        Mat roi_face = source_image(Rect(origin, area));//For detect eyes.
        int eye_factor = 16;
        int eye_flag = CASCADE_FIND_BIGGEST_OBJECT;
        min_eyes = Size(faces[i].width/eye_factor, faces[i].height/eye_factor);
        eyes_cascade.detectMultiScale(roi_face, eyes,scale_factor, min_neighbors, eye_flag, min_eyes);

        cout << "face:" << (i+1) << " eyes:" << eyes.size() ;
        if (clog.is_open()) clog << "face:" << (i+1) << " eyes:" << eyes.size() ;

        if(eyes.size() > 0){
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

            //Get high frequency area from DFT result.
            const int crop = 8;
            Point origin(magI.cols/crop, magI.rows/crop);
            Size area(magI.cols - (magI.cols/crop), magI.rows - (magI.cols/crop));
            Mat hf_area = magI(Rect(origin,area));

            double hf_avg = mean(hf_area)[0];
            if (max_avg < hf_avg) {
                max_avg = hf_avg;
            }
            cout << " avg:" << hf_avg;
            if (clog.is_open()) clog << "avg:" << hf_avg;

            normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
            // viewable image form (float between values 0 and 1).
        }
        cout << endl;
        if (clog.is_open()) clog << endl;
    }
    //Output result
    int accuracy = (int)((max_avg * 10.0) + 0.5);
    cout << "accuracy:" << accuracy << endl;
    if (clog.is_open()) clog << "accuracy:" << accuracy << endl;
    exit(accuracy);
}
