#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

const String prototxt_path = "/path/to/your_project_folder_directory/Models/MobileNetSSD_deploy.prototxt"; // change /path/to/your_project_folder_directory to your actuall directory
const String model_path = "/path/to/your_project_folder_directory/Models/MobileNetSSD_deploy.caffemodel"; // change /path/to/your_project_folder_directory to your actuall directory
const float confidence_threshold = 0.01;
const vector<string> classes = { "background", "aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                  "sofa", "train", "tvmonitor" };


int main(int argc, char* argv[]) {

    if(argc < 2){
        std::cerr << "Error Missing Arguments!\nUsage: ./person_detect <arguments>\n--> where the <arguments> is the camera index like 1 or 0 or 2" << std::endl;
        return -1;
    }

    int cameraIndex = std::stoi(argv[1]);

    VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        cerr << "Error: Couldn't open the camera." << endl;
        return -1;
    }

    auto fps_start_time = chrono::high_resolution_clock::now();
    double fps = 0.0;
    int total_frames = 0;
    double scale = 5.0;
    bool detected = false;
    int sensor = 0;
    Net detector = readNetFromCaffe(prototxt_path, model_path);

    while (true) {
        Mat frame;
        cap.read(frame);

        Mat secondFrame = frame.clone();
        resize(secondFrame, secondFrame, Size(secondFrame.size().width / scale, secondFrame.size().height / scale));

        total_frames++;

        int H = secondFrame.rows;
        int W = secondFrame.cols;

        Mat blob = blobFromImage(secondFrame, 0.007843, Size(W, H), Scalar(127.5), false);

        detected = false;
        
        detector.setInput(blob);
        Mat person_detections = detector.forward();

        for (int i = 0; i < person_detections.size[2]; i++) {
            float confidence = person_detections.ptr<float>(0, 0, i)[2];

            if (confidence > confidence_threshold) {
                int idx = static_cast<int>(person_detections.ptr<float>(0, 0, i)[1]);

                if (classes[idx] != "person") {
                    continue;
                }

                float* data = person_detections.ptr<float>(0, 0, i);

                float startX = data[3] * W;
                float startY = data[4] * H;
                float endX = data[5] * W;
                float endY = data[6] * H;

                detected = true;

                Rect person_box(static_cast<int>(startX * scale), static_cast<int>(startY * scale), static_cast<int>((endX * scale) - (startX * scale)), static_cast<int>((endY * scale) - (startY * scale)));
                rectangle(frame, person_box, Scalar(0, 0, 255), 2);
            }
        }

        // FPS SIDE
        auto fps_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> time_diff = fps_end_time - fps_start_time;
        if (time_diff.count() == 0) {
            fps = 0.0;
        } else {
            fps = total_frames / time_diff.count();
        }

        string fps_text = "FPS: " + to_string(fps);

        putText(frame, fps_text, Point(5, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);

        if(detected && sensor != 100){
            cout << "Person Detected!" << endl;
            sensor++;
        } else if(sensor != 0 && !detected){
            cout << "Person Not Detected!" << endl;
            sensor--;
        }

        if(sensor == 0){
            cout << "Person Not Detected!" << endl;
        }

        if(sensor == 100){
            cout << "Person Detected!" << endl;
        }

        cout << "Sensor: " << sensor << endl;
        std::cout << "\033[2J]\033[1;1H";

        imshow("Camera Frame", frame);

        char key = static_cast<char>(waitKey(1));
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
