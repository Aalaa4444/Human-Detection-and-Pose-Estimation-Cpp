#include <iostream>
#include <cstdlib>
#include <cstdio>

#include <cstring>
#include <fstream>
#include <sstream>

#include <cmath> 
#include <random> 

#include <opencv2/core/core.hpp> 
#include <opencv2/video/video.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/videoio/videoio.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/imgcodecs/imgcodecs.hpp> 

#include <opencv2/dnn/dnn.hpp>


using namespace std;
using namespace cv;
using namespace dnn;

vector<string> class_names = { "person" };
RNG rng(3);

Mat prepare_input(Mat image, int input_width, int input_height) {
    Mat blob;
    dnn::blobFromImage(image, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    return blob;
}

vector<Rect> rescale_boxes(vector<Rect> boxes, int img_width, int img_height) {
    vector<Rect> rescaled_boxes;
    float scale_x = static_cast<float>(img_width) / 640.0;
    float scale_y = static_cast<float>(img_height) / 640.0;

    for (const auto& box : boxes) {
        int x1 = static_cast<int>(box.x * scale_x);
        int y1 = static_cast<int>(box.y * scale_y);
        int x2 = static_cast<int>((box.x + box.width) * scale_x);
        int y2 = static_cast<int>((box.y + box.height) * scale_y);

        rescaled_boxes.emplace_back(Rect(x1, y1, x2 - x1, y2 - y1));
    }

    return rescaled_boxes;
}

vector<Rect> xywh2xyxy(vector<Rect> boxes) {
    vector<Rect> converted_boxes;
    for (auto& box : boxes) {
        int x1 = box.x - box.width / 2;
        int y1 = box.y - box.height / 2;
        int x2 = box.x + box.width / 2;
        int y2 = box.y + box.height / 2;
        converted_boxes.emplace_back(Rect(x1, y1, x2 - x1, y2 - y1));
    }
    return converted_boxes;
}

void drawdetect(Mat& image, vector<Rect> boxes, vector<float> scores, vector<int> class_ids, float mask_alpha = 0.3) {
    Mat det_img = image.clone();
    int img_height = image.rows;
    int img_width = image.cols;
    float font_size = min(img_height, img_width) * 0.0006;
    int text_thickness = static_cast<int>(min(img_height, img_width) * 0.001);

    cout << "Total of detections = " << boxes[0] << endl;

    for (size_t i = 0; i < boxes.size(); ++i) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        rectangle(det_img, boxes[i], color, 2);

        // std::string label = class_names[class_ids[i]] + " " + std::to_string(static_cast<int>(scores[i] * 100)) + "%";
        Point org(boxes[i].tl().x, boxes[i].tl().y - 10);
        putText(det_img, "hola", org, FONT_HERSHEY_SIMPLEX, font_size, Scalar(255, 255, 255), text_thickness);
    }

    imshow("Output", det_img);
    waitKey(0);
}

struct Detection
{
    int class_id{ 0 };
    string className{};
    float confidence{ 0.0 };
    Scalar color{};
    Rect box{};
};

int main() {
    string model_path = "D:/blackboard/AI_online/Advanced_AI/conv/yolov8n.onnx";
    float modelNMSThreshold = 0.05;
    float modelScore = 0.1;

    // Initialize model
    dnn::Net net = dnn::readNetFromONNX(model_path);

    // Load image
    string img_url = "D:/blackboard/AI_online/Advanced_AI/conv/test.jpg";
    Mat img = imread(img_url);

    // Prepare input
    Mat input_tensor = prepare_input(img, 640, 640);

    // Perform inference
    net.setInput(input_tensor);
    vector<int> output_names;
    output_names.push_back(267);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);

    transpose(outputs[0], outputs[0]);

    float* data = (float*)outputs[0].data;

    float x_factor = static_cast<float>(img.cols) / 640;
    float y_factor = static_cast<float>(img.rows) / 640;

    // Process output
    vector<Rect> boxes;
    vector<float> conf;
    vector<int> class_ids;

    for (int i = 0; i < rows; ++i) {
        float* classes_scores = data + 4;

        Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
        Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScore) {
            cout << "Number of detections:" << maxClassScore << endl;
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(Rect(left, top, width, height));
            conf.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
        }

        data += dimensions;
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, conf, modelScore, modelNMSThreshold, nms_result);

    vector<Detection> detections{};

    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = conf[idx];

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(100, 255);
        result.color = Scalar(dis(gen), dis(gen), dis(gen));

        result.className = class_names[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    int dt = detections.size();
    cout << "Number of detections:" << dt << endl;

    Mat frame = img.clone();
    for (int i = 0; i < dt; ++i)
    {
        Detection New_detection = detections[i];

        Rect box = New_detection.box;
        cout << "Number of detections:" << box << endl;
        Scalar color = New_detection.color;

        // Detection box
        cout << box;
        rectangle(frame, box, color, 1);

        // Detection box text
        string classString = New_detection.className + ' ' + to_string(New_detection.confidence).substr(0, 4);
        Size textSize = getTextSize(classString, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        rectangle(frame, textBox, color, FILLED);
        putText(frame, classString, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }

    imshow("Inference", frame);
    imwrite("output.jpg", frame);

    waitKey(0);

    return 0;
}
