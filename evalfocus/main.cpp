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

//extern char *optarg;
//extern int optind, opterr;

int main(int argc, char * argv[]) {
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    vector<Rect> faces;
    vector<Rect> eyes;
    Size min_face;
    Size min_eyes;
    int min_neighbors = 3;
    double scale_factor = 1.15;

    Mat source_image;
    string source_file;
    string face_cascade_file;
    string eyes_cascade_file;
    int op;
    opterr = 0;
    while ( (op = getopt(argc,argv,"f:c:")) != -1){
        switch (op){
            case 'f':
                source_file = optarg;
                break;
            case 'c':
                face_cascade_file = optarg;
        }
    }
    
    if (source_file.empty()){
        source_file = argv[1];
    }
    if (face_cascade_file.empty()){
        face_cascade_file = default_face_cascade_file;
    }
    if (!face_cascade.load(face_cascade_file)){
        cerr << "face cascade file not found." << endl;
        exit(1);
    }

    eyes_cascade_file = default_eyes_cascade_file;
    if (!eyes_cascade.load(eyes_cascade_file)){
        cerr << "eyes cascade file not found." << endl;
        exit(1);
    }

    source_image = imread(source_file,IMREAD_GRAYSCALE);
    int longside = max(source_image.size().width,source_image.size().height);
    int rectsize = longside / 16;
    min_face = Size(rectsize,rectsize);
//  imshow("source",source_image);
    
    face_cascade.detectMultiScale(source_image, faces, scale_factor, min_neighbors, CASCADE_FIND_BIGGEST_OBJECT , min_face);
    cout << "detected faces: "<< faces.size() << endl;
    
    Mat roi_face;
    for (int i = 0; i < faces.size(); i++){
        Point origin(faces[i].x,faces[i].y);
        Size area(faces[i].width,faces[i].height);
        roi_face = source_image(Rect(origin, area));
        int eye_factor = 16;
        min_eyes = Size(faces[i].width / eye_factor, faces[i].height / eye_factor);
        eyes_cascade.detectMultiScale(roi_face, eyes,scale_factor,min_neighbors,CASCADE_FIND_BIGGEST_OBJECT,min_eyes);
        cout << "face: " << (i+1) << " eyes: " << eyes.size() ;
        if(eyes.size() > 0){
#if 0
            //draw rectangle
            rectangle(source_image, Point(faces[i].x,faces[i].y),Point(faces[i].x + faces[i].width,faces[i].y + faces[i].height),Scalar(200,200,0),1,CV_AA);
            for (int j = 0; j < eyes.size(); j++){
                rectangle(roi_face, Point(eyes[j].x,eyes[j].y),Point(eyes[j].x + eyes[j].width,eyes[j].y + eyes[j].height),Scalar(0,200,0),2,CV_AA);
            }
#endif
            int optx = getOptimalDFTSize( roi_face.rows );
            int opty = getOptimalDFTSize( roi_face.cols );
            Mat padded_face;
            copyMakeBorder(roi_face, padded_face, 0, optx - roi_face.rows, 0, opty - roi_face.cols, BORDER_CONSTANT, Scalar::all(0));
            
            
            Mat planes[] = {Mat_<float>(padded_face), Mat::zeros(padded_face.size(), CV_32F)};
            Mat complexI;
            merge(planes, 2, complexI);  // Add to the expanded another plane with zeros
            dft(complexI, complexI);     // this way the result may fit in the source matrix
            split(complexI, planes);     // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
            magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
            Mat magI = planes[0];

            magI += Scalar::all(1);                    // switch to logarithmic scale
            log(magI, magI);
            // crop the spectrum, if it has an odd number of rows or columns
            magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
#if 0
            // rearrange the quadrants of Fourier image  so that the origin is at the image center
            int cx = magI.cols/2;
            int cy = magI.rows/2;
            Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
            Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
            Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
            Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
            Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
            q2.copyTo(q1);
            tmp.copyTo(q2);
#endif
            int crop = 8;
            Point origin(magI.cols/crop,magI.rows/crop);
            Size area(magI.cols - (magI.cols/crop), magI.rows - (magI.cols/crop));
            Mat hf_area = magI(Rect(origin,area));
            cout << " avg: " << (int)( (mean(hf_area)[0] * 10.0) + 0.5 );
            
            normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
            // viewable image form (float between values 0 and 1).
//            imshow("Input Image", roi_face);    // Show the result
//            imshow("spectrum magnitude", magI);
//            waitKey();
        }
        cout << endl;
    }
}
