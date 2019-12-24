#include "yolov3-new.hpp"

//using json = nlohmann::json;
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

/*全局变量*/

float sigmoid(float in) {
    return 1.f / (1.f + exp(-in));
}

float exponential(float in) {
    return exp(in);
}

int yolov3::loadYaml(string ymlFilename) {
    YAML::Node root = YAML::LoadFile(ymlFilename);
    YAML::Node config = root["config"];
    if (!config.IsDefined()) {
        std::cout << "config is null" << std::endl;
        return 0;
    }
    if (!config.IsMap()) {
        std::cout << "config is not map" << std::endl;
        return 0;
    }
    onnxFile = config["onnxFile"].as<std::string>();
    engineFile = config["engineFile"].as<std::string>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    idx = config["idx"].as<int>();
    CATEGORY = config["CATEGORY"].as<int>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    DETECT_WIDTH = config["DETECT_WIDTH"].as<int>();
    DETECT_WIDTH = config["DETECT_HEIGHT"].as<int>();

    output_shape_1 = config["output_shape_1"].as<int>();
    output_shape_2 = config["output_shape_2"].as<int>();
    output_shape_3 = config["output_shape_3"].as<int>();

    auto id2namelist = config["id2name"];
    for (int i = 0; i < CATEGORY; ++i) {
        if (id2namelist[i].IsDefined()) {
            id2name[id2namelist[i]["id"].as<int>()]=id2namelist[i]["name"].as<std::string>();
        } else {
            continue;
        }
    }

    auto g_anchors_list = config["g_anchors"];
    vector<vector<float>> g_anchors;
    for (int j = 0; j < 9 ; ++j) {
        if(g_anchors_list[j].IsDefined()){
            g_anchors.push_back({g_anchors_list[j]["x"].as<int>(),g_anchors_list[j]["y"].as<int>()});
        }else{
            continue;
        }
    }
}

float *yolov3::merge(float *out1, float *out2, float *out3, int bsize_out1, int bsize_out2, int bsize_out3) {
    float *out_total = new float[bsize_out1 + bsize_out2 + bsize_out3];

    for (int j = 0; j < bsize_out1; ++j) {
        int index = j;
        out_total[index] = out1[j];
    }

    for (int j = 0; j < bsize_out2; ++j) {
        int index = j + bsize_out1;
        out_total[index] = out2[j];
    }

    for (int j = 0; j < bsize_out3; ++j) {
        int index = j + bsize_out1 + bsize_out2;
        out_total[index] = out3[j];
    }
    return out_total;
}

void yolov3::DoNms(vector<DetectionRes> &detections, float nmsThresh) {
    auto iouCompute = [](float *lbox, float *rbox) {
        float interBox[] = {
                max(lbox[0], rbox[0]), //left
                min(lbox[0] + lbox[2], rbox[0] + rbox[2]), //right
                max(lbox[1], rbox[1]), //top
                min(lbox[1] + lbox[3], rbox[1] + rbox[3]), //bottom
        };

        if (interBox[2] >= interBox[3] || interBox[0] >= interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0] + 1) * (interBox[3] - interBox[2] + 1);
        return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
    };

    sort(detections.begin(), detections.end(), [=](const DetectionRes &left, const DetectionRes &right) {
        return left.prob > right.prob;
    });

    vector<DetectionRes> result;
    std::cout << "nms starting:" << endl;
    for (unsigned int m = 0; m < detections.size(); m++) {
        printf("%d, %f, %f, %f, %f, %f\n", detections[m].classes,
               detections[m].x, detections[m].y, detections[m].w, detections[m].h, detections[m].prob);
    }
    std::cout << "nms end:" << endl;
    for (unsigned int m = 0; m < detections.size(); ++m) {
        result.push_back(detections[m]);
        int class_number = detections[m].classes;
        for (unsigned int n = m + 1; n < detections.size(); ++n) {
            if (class_number == detections[n].classes and
                iouCompute((float *) (&detections[m]), (float *) (&detections[n])) > nmsThresh) {
                detections.erase(detections.begin() + n);
                --n;
            }
        }
    }
    detections = move(result);
}

vector<DetectionRes> yolov3::postProcess(cv::Mat &image, float *output) {
    vector<DetectionRes> detections;
    int total_size = 0;
    for (unsigned int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];
        int size = 1;
        for (unsigned int j = 0; j < shape.size(); j++) {
            size *= shape[j];
        }
        total_size += size;
    }

    int offset = 0;
    float *transposed_output = new float[total_size];
    float *transposed_output_t = transposed_output;
    for (unsigned int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];  // nchw
        int chw = shape[1] * shape[2] * shape[3];
        int hw = shape[2] * shape[3];
        for (int n = 0; n < shape[0]; n++) {
            int offset_n = offset + n * chw;
            for (int h = 0; h < shape[2]; h++) {
                for (int w = 0; w < shape[3]; w++) {
                    int h_w = h * shape[3] + w;
                    for (int c = 0; c < shape[1]; c++) {
                        int offset_c = offset_n + hw * c + h_w;
                        *transposed_output_t++ = output[offset_c];
                    }
                }
            }
        }
        offset += shape[0] * chw;
    }
    vector<vector<int>> shapes;
    for (unsigned int i = 0; i < output_shape.size(); i++) {
        auto shape = output_shape[i];
        vector<int> tmp = {shape[2], shape[3], 3, (CATEGORY + 5)};
        shapes.push_back(tmp);
    }

    offset = 0;
    for (unsigned int i = 0; i < output_shape.size(); i++) {
        auto masks = g_masks[i];
        vector<vector<float>> anchors;
        for (auto mask : masks)
            anchors.push_back(g_anchors[mask]);

        std::cout << "output_shape size: " << output_shape.size() << endl;
        auto shape = shapes[i];
        std::cout << "shape: " << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << endl;

        for (int h = 0; h < shape[0]; h++) {
            int offset_h = offset + h * shape[1] * shape[2] * shape[3];
            for (int w = 0; w < shape[1]; w++) {
                int offset_w = offset_h + w * shape[2] * shape[3];
                for (int c = 0; c < shape[2]; c++) {
                    int offset_c = offset_w + c * shape[3];
                    float *ptr = transposed_output + offset_c;

                    int max_index = -1;
                    float max_value = FLT_MIN;
                    for (int k = 5; k < CATEGORY + 5; k++) {
                        if (ptr[k] > max_value) {
                            max_value = ptr[k];
                            max_index = k - 5;
                        }
                    }
                    float score = sigmoid(max_value) * sigmoid(ptr[4]);

                    if (score < obj_threshold)
                        continue;
                    ptr[0] = sigmoid(ptr[0]);
                    ptr[1] = sigmoid(ptr[1]);
                    ptr[2] = exponential(ptr[2]) * anchors[c][0];
                    ptr[3] = exponential(ptr[3]) * anchors[c][1];

                    ptr[0] += w;
                    ptr[1] += h;
                    ptr[0] /= shape[0];
                    ptr[1] /= shape[1];
                    ptr[2] /= DETECT_WIDTH;
                    ptr[3] /= DETECT_HEIGHT;
                    ptr[0] -= ptr[2] / 2;
                    ptr[1] -= ptr[3] / 2;
                    std::cout << "未处理2： " << ptr[0] << " " << ptr[1] << " " << ptr[2] << " " << ptr[3] << endl;
                    DetectionRes det;
                    det.x = ptr[0];
                    det.y = ptr[1];
                    det.w = ptr[2];
                    det.h = ptr[3];
                    det.classes = max_index;
                    det.prob = score;
                    std::cout << "未处理3： " << det.x << " " << det.y << " " << det.w << " " << det.h << endl;
                    detections.push_back(det);
                }
            }
        }
        offset += shape[0] * shape[1] * shape[2] * shape[3];
    }
    delete[]transposed_output;

    float h = DETECT_HEIGHT;   //net h
    float w = DETECT_WIDTH;   //net w

    //scale bbox to img
    int width = image.cols;
    int height = image.rows;
    float scale[] = {float(w) / width, float(h) / height};
    float scaleSize[] = {DETECT_WIDTH, DETECT_HEIGHT};

    //correct box
    for (auto &bbox : detections) {
        bbox.x = (bbox.x * w - (w - scaleSize[0]) / 2.f) / scale[0];
        bbox.y = (bbox.y * h - (h - scaleSize[1]) / 2.f) / scale[1];
        bbox.w *= w;
        bbox.h *= h;
        bbox.w /= scale[0];
        bbox.h /= scale[1];
        std::cout << "未处理4： " << bbox.x << " " << bbox.y << " " << bbox.w << " " << bbox.h << endl;
    }

    //nms
    float nmsThresh = nms_threshold;
    if (nmsThresh > 0)
        DoNms(detections, nmsThresh);

    return detections;
}


// prepare img
vector<float> yolov3::prepareImage(vector<cv::Mat> imgs) {
    auto scaleSize = cv::Size(DETECT_WIDTH, DETECT_HEIGHT);
    vector<float> result(BATCH_SIZE * DETECT_HEIGHT
                         * DETECT_WIDTH * INPUT_CHANNEL);
    cv::Mat resized;
    auto data = result.data();
    for (auto iter = imgs.begin(); iter != imgs.end(); iter++) {
        auto img = *iter;
        std::cout << "before resize" << endl;
        std::cout << img.cols << " " << img.rows << endl;
        std::cout << "DETECT_WIDTH:" << DETECT_WIDTH << " DETECT_HEIGHT:" << DETECT_HEIGHT << endl;
        cv::resize(img, resized, scaleSize, 0, 0, INTER_CUBIC);
        std::cout << "after resize" << endl;

        cv::Mat img_float;
        resized.convertTo(img_float, CV_32FC3, 1.f / 255.0);
        //HWC TO CHW
        vector<Mat> input_channels(INPUT_CHANNEL);
        cv::split(img_float, input_channels);
        int channelLength = DETECT_HEIGHT * DETECT_WIDTH;
        for (int i = 0; i < INPUT_CHANNEL; ++i) {
            memcpy(data, input_channels[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}


// load engine file
bool yolov3::readTrtFile(const std::string &engineFile, //name of the engine file
                         IHostMemory *&trtModelStream)  //output buffer for the TensorRT model
{
    using namespace std;
    fstream file;
    cout << "loading filename from:" << engineFile << endl;
    nvinfer1::IRuntime *trtRuntime;
    nvonnxparser::IPluginFactory *onnxPlugin = createPluginFactory(gLogger);
    file.open(engineFile, ios::binary | ios::in);
    file.seekg(0, ios::end);
    int length = file.tellg();
    file.seekg(0, ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    cout << "load engine done" << endl;
    std::cout << "deserializing" << endl;
    trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    ICudaEngine *engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    cout << "deserialize done" << endl;
    trtModelStream = engine->serialize();

    return true;
}


void yolov3::onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                            const std::string &filename,  // name of saved engine
                            IHostMemory *&trtModelStream) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition *network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(yolov3::BATCH_SIZE);
    builder->setMaxWorkspaceSize(1 << 27);
    builder->setFp16Mode(true);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8) {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    //	samplesCommon::enableDLA(builder, gArgs.useDLACore);
    cout << "start building engine" << endl;
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    cout << "build engine done" << endl;
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine
    trtModelStream = engine->serialize();

    // save engine
    nvinfer1::IHostMemory *data = engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    cout << "writing engine file..." << endl;
    file.write((const char *) data->data(), data->size());
    cout << "save engine file done" << endl;
    file.close();

    // then close everything down
    engine->destroy();
    network->destroy();
    builder->destroy();
}

int64_t yolov3::volume(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

unsigned int yolov3::getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

void yolov3::prepareEngine() {
    // create a TensorRT model from the onnx model and serialize it to a stream
    trtModelStream = nullptr;
    loadYaml("yolov3.yml");

    // create and load engine
    fstream existEngine;
    existEngine.open(engineFile, ios::in);
    if (existEngine) {
        readTrtFile(engineFile, trtModelStream);
        assert(trtModelStream != nullptr);
    } else {
        onnxToTRTModel(onnxFile, engineFile, trtModelStream);
        assert(trtModelStream != nullptr);
    }

    //get engine
    assert(trtModelStream != nullptr);
    runtime = createInferRuntime(gLogger);
    onnxPlugin = createPluginFactory(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0) {
        runtime->setDLACore(gArgs.useDLACore);
    }
    engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), onnxPlugin);

    //get context
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 4);

    nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = BATCH_SIZE * volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        cout << "binding" << i << ": " << totalSize << endl;
        CHECK(cudaMalloc(&buffers[i], totalSize));
    }

    //get stream
    CHECK(cudaStreamCreate(&stream));
    //define inputImgs inputData outputDetections ...
    outSize1 = bufferSize[1] / sizeof(float);
    outSize2 = bufferSize[2] / sizeof(float);
    outSize3 = bufferSize[3] / sizeof(float);
    out1 = new float[outSize1];
    out2 = new float[outSize2];
    out3 = new float[outSize3];
}

vector<output> yolov3::doInferenceFrieza(vector<cv::Mat> inputImgs) {

    outputs.clear();
    result_images.clear();

    auto t_start_pre = std::chrono::high_resolution_clock::now();
    vector<float> all_Input = prepareImage(inputImgs);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "prepare image take: " << total_pre << " ms." << endl;

    if (!all_Input.data()) {
        cout << "prepare images ERROR!" << endl;
    }

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    std::cout << "host2device" << endl;
    CHECK(cudaMemcpyAsync(buffers[0], all_Input.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));

    // do inference
    std::cout << "execute" << endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(BATCH_SIZE, buffers);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total << " ms." << endl;
    std::cout << "execute success" << endl;
    std::cout << "device2host" << endl;
    CHECK(cudaMemcpyAsync(out1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    std::cout << "copy cuda memory success" << endl << endl;
    std::cout << "device2host sueccess" << endl << endl;
    std::cout << "merge data" << endl;
    int offset_out1 = 0;
    int offset_out2 = 0;
    int offset_out3 = 0;
    for (auto iter = inputImgs.begin(); iter != inputImgs.end(); iter++) {
        auto img = *iter;
        float *out = merge(out1 + offset_out1, out2 + offset_out2, out3 + offset_out3,
                           outSize1 / BATCH_SIZE, outSize2 / BATCH_SIZE, outSize3 / BATCH_SIZE);
        std::cout << "merge data success" << endl << endl;
        // postprocess
        std::cout << "postProcess data" << endl;
        auto t_start_post = std::chrono::high_resolution_clock::now();
        auto boxes = postProcess(img, out);
        auto t_end_post = std::chrono::high_resolution_clock::now();
        float total_post = std::chrono::duration<float, std::milli>(t_end_post - t_start_post).count();
        std::cout << "Postprocess take: " << total_post << " ms." << endl;
        std::cout << "postProcess data success" << endl << endl;
        //print boxes
        for (unsigned int i = 0; i < boxes.size(); ++i) {
            cout << boxes[i].prob << ", " << boxes[i].x << ", " << boxes[i].y << ", " << boxes[i].w << ", "
                 << boxes[i].h << endl;
        }

        outputs.push_back(boxes);

        cout << "\n" << endl;
        offset_out1 += output_shape_1 * output_shape_1 * INPUT_CHANNEL * (CATEGORY + 5);
        offset_out2 += output_shape_2 * output_shape_2 * INPUT_CHANNEL * (CATEGORY + 5);
        offset_out3 += output_shape_3 * output_shape_3 * INPUT_CHANNEL * (CATEGORY + 5);
    }

    //保存结果 并返回
    auto iterDet = outputs.begin();
    for (unsigned int i = 0; i < inputImgs.size(); ++i, ++iterDet) {
        const vector<DetectionRes> &outputI = *iterDet;
        output tmpoutput;
        for (auto box : outputI) {
            finalRes tmpfinalRes;
            tmpfinalRes.left_x = box.x;
            tmpfinalRes.left_y = box.y;
            tmpfinalRes.right_x = box.x + box.w;
            tmpfinalRes.right_y = box.y + box.h;
            tmpfinalRes.confidence = box.prob;
            auto classname = id2name.find(box.classes);
            if (classname != id2name.end()) {
                tmpfinalRes.class_name = classname->second;
            } else {
                tmpfinalRes.class_name = "UNO";
            }
            tmpoutput.boxes.push_back(tmpfinalRes);
        }
        tmpoutput.frameID = idx;
        idx++;
        result_images.push_back(tmpoutput);
    }

    return result_images;
}

void yolov3::releaseEngine() {
    // release the stream and the buffers
    capture.release();
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaFree(buffers[2]));
    CHECK(cudaFree(buffers[3]));

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int yolov3::readVideo(SyncQueue<output> *queuebuffer, string filename) {
    capture.open(filename);

    if (!capture.isOpened()) {
        std::cout << "Read video Failed !" << std::endl;
        return 0;
    }

    int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "total frame number is: " << frame_num << std::endl;
    vector<cv::Mat> batch;
    vector<shared_ptr<cv::Mat>> batch_ptr;

    for (int frameId = 0; frameId < frame_num - 1; ++frameId) {
        std::shared_ptr<cv::Mat> frame = make_shared<cv::Mat>();
        capture >> *frame;
        batch.push_back(*frame);
        batch_ptr.push_back(frame);
        if (batch.size() % 10 == 0) {
            auto content = doInferenceFrieza(batch);
            /*将content存入queue*/
            auto i = 0;
            while (content.size() > 0) {
                auto tmp_data = content.front();
                tmp_data.image_ptr = batch_ptr[i];
                (*queuebuffer).Put(tmp_data);
                content.erase(content.begin());
                i++;
            }
            /*将content存入queue*/
            batch.clear();
            batch_ptr.clear();
        }
    }
    if (batch.size() > 0) {
        auto content = doInferenceFrieza(batch);
        auto i = 0;
        while (content.size() > 0) {
            auto tmp_data = content.front();
            tmp_data.image_ptr = batch_ptr[i];
            (*queuebuffer).Put(tmp_data);
            content.erase(content.begin());
            i++;
        }
        batch.clear();
        batch_ptr.clear();
    }
    return 1;
}


