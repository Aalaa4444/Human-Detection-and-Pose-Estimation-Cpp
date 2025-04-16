#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

const int POSE_PAIRS[17][2] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4},
    {5, 6}, {6, 7}, {1, 8}, {8, 9},
    {9, 10}, {1, 11}, {11, 12}, {12, 13},
    {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}
};

const int NUM_POINTS = 18;

int main()
{
    String protoFile = "pose_deploy_linevec.prototxt";
    String weightsFile = "pose_iter_440000.caffemodel";
    Net net = readNetFromCaffe(protoFile, weightsFile);

    Mat frame = imread("D:/blackboard/AI_online/Advanced_AI/conv/test.jpg");
    if (frame.empty()) {
        cerr << "Image not found!\n";
        return -1;
    }

    int inWidth = 368;
    int inHeight = 368;
    float thresh = 0.1;

    Mat inputBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
    net.setInput(inputBlob);
    Mat output = net.forward();

    int H = output.size[2];
    int W = output.size[3];

    vector<Point> points(NUM_POINTS);

    for (int n = 0; n < NUM_POINTS; ++n) {
        Mat probMap(H, W, CV_32F, output.ptr(0, n));
        Mat probMapResized;
        resize(probMap, probMapResized, frame.size());

        Point maxLoc;
        double prob;
        minMaxLoc(probMapResized, 0, &prob, 0, &maxLoc);

        if (prob > thresh) {
            circle(frame, maxLoc, 5, Scalar(0, 255, 255), -1);
            putText(frame, to_string(n), maxLoc, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            points[n] = maxLoc;
        }
        else {
            points[n] = Point(-1, -1);
        }
    }

    for (int i = 0; i < 17; ++i) {
        Point2f partA = points[POSE_PAIRS[i][0]];
        Point2f partB = points[POSE_PAIRS[i][1]];

        if (partA.x > 0 && partB.x > 0)
            line(frame, partA, partB, Scalar(0, 255, 0), 2);
    }

    imshow("Pose Estimation", frame);
    imwrite("output.jpg", frame);
    waitKey(0);
    return 0;
}
