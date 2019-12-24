//
// Created by huangwei01 on 2019/10/12.
//

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"
#include "argsParser.h"
#include "common.h"
#include "logger.h"
#include "queue.cpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <float.h>
#include <yaml-cpp/yaml.h>

#ifndef CPP_PROJECT_YOLOV3_TRT_H
#define CPP_PROJECT_YOLOV3_TRT_H
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

typedef struct DetectionRes {
    int classes;
    float x;
    float y;
    float w;
    float h;
    float prob;
} DetectionRes;

typedef struct finalRes {
    string class_name;  //目标类别，通过class_name选择DCL或者WPAL进行处理
    float left_x;
    float left_y;
    float right_x;
    float right_y;
    float confidence;   //最佳抓拍图像置信度
} finalRes;

typedef struct output {
    int frameID;
    shared_ptr<cv::Mat> image_ptr;
    vector<finalRes> boxes; //key是类  value是boxes
} output;

//functions

extern "C" {

}

class yolov3 {
public:
    /*全局变量*/
    IRuntime *runtime;
    nvonnxparser::IPluginFactory *onnxPlugin;
    ICudaEngine *engine;
    IExecutionContext *context;
    IHostMemory *trtModelStream;
    list<vector<DetectionRes>> outputs;
    std::vector<int64_t> bufferSize;
    vector<output> result_images;
    cv::VideoCapture capture;
    samplesCommon::Args gArgs;

    int nbBindings;
    void *buffers[4];
    float *out1;
    float *out2;
    float *out3;
    int outSize1;
    int outSize2;
    int outSize3;
    cudaStream_t stream;


    //these parts need to be initialized

    int idx = 1;
    string onnxFile = "yolov3.onnx";
    string engineFile = "yolov3.trt";

    float obj_threshold = 0.60;
    float nms_threshold = 0.5;

    int CATEGORY = 80;
    int BATCH_SIZE = 2;
    int INPUT_CHANNEL = 3;
    float DETECT_WIDTH = 608;
    float DETECT_HEIGHT = 608;

    int output_shape_1 = 19;
    int output_shape_2 = 38;
    int output_shape_3 = 76;

    vector<vector<int>> output_shape = {{1, (CATEGORY + 5) * 3, output_shape_1, output_shape_1},
                                        {1, (CATEGORY + 5) * 3, output_shape_2, output_shape_2},
                                        {1, (CATEGORY + 5) * 3, output_shape_3, output_shape_3}};

    vector<vector<int>> g_masks = {{6, 7, 8},
                                   {3, 4, 5},
                                   {0, 1, 2}};

    vector<vector<float>> g_anchors = {{10,  13},
                                       {16,  30},
                                       {33,  23},
                                       {30,  61},
                                       {62,  45},
                                       {59,  119},
                                       {116, 90},
                                       {156, 198},
                                       {373, 326}};

    std::unordered_map<int, std::string> id2name = {{0, "Canberra"},
                                                    {1, "Washington"},
                                                    {2, "Paris"}};

public:

    void prepareEngine();

    void releaseEngine();

    int readVideo(SyncQueue<output> *queuebuffer, string filename);

private:
    int loadYaml(string ymlFilename);

    float *merge(float *out1, float *out2, float *out3, int bsize_out1, int bsize_out2, int bsize_out3);

    void DoNms(vector<DetectionRes> &detections, float nmsThresh);

    vector<DetectionRes> postProcess(cv::Mat &image, float *output);

    vector<float> prepareImage(vector<cv::Mat> imgs);

    void onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                        const std::string &filename,  // name of saved engine
                        IHostMemory *&trtModelStream);

    int64_t volume(const nvinfer1::Dims &d);

    unsigned int getElementSize(nvinfer1::DataType t);

    vector<output> doInferenceFrieza(vector<cv::Mat> inputImgs);

    bool readTrtFile(const std::string &engineFile, //name of the engine file
                     IHostMemory *&trtModelStream);
};


#endif //CPP_PROJECT_YOLOV3_TRT_H
