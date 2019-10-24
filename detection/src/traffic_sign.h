//
// Created by pengpeng on 2019/10/17.
//

#ifndef SRC_TRAFFIC_SIGN_H
#define SRC_TRAFFIC_SIGN_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

typedef struct ObjRect {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
}ObjRect;

typedef struct ObjInfo {
    ObjRect bbox;
    cv::Vec4f regression;
}ObjInfo;

class CascadeCNN {
public:
    CascadeCNN(const std::string& model_dir, const bool use_gpu = false);
    void Detect(const cv::Mat& img, std::vector<ObjInfo>& ObjInfo, int minSize, double* threshold, double factor);


private:
    void img2tensor(cv::Mat & img, at::Tensor & tensor);
    void tensor2img(at::Tensor tensor, cv::Mat & img);
    at::Tensor get_output(at::Tensor input_tensor);
    at::Tensor get_argmax(at::Tensor input_tensor);

    std::vector<ObjInfo> NonMaximumSuppression(std::vector<ObjInfo>& bboxes, float thresh, char methodType);
    void Bbox2Square(std::vector<ObjInfo>& bboxes);
    std::vector<ObjInfo> BoxRegress(std::vector<ObjInfo>& objInfo,int stage);
    void Padding(int img_w,int img_h);

    void GenerateBoundingBox(at::Tensor cls, at::Tensor reg, float scale, float threshold, const int ws, const int hs);
    void ClassifyObj_batch(const std::vector<ObjInfo>& regressed_rects,cv::Mat &sample_single,
                           torch::jit::script::Module& net,double thresh,char netName);
    void plot(cv::Mat& img, std::vector<ObjInfo>& ObjInfo);

private:
    torch::jit::script::Module PNet_;
    torch::jit::script::Module RNet_;
//    torch::jit::script::Module Classifier;
    std::vector<ObjInfo> candidate_rects_;
    std::vector<ObjInfo> total_boxes_;
    std::vector<ObjInfo> regressed_rects_;
    std::vector<ObjInfo> regressed_padding_;

    std::vector<cv::Mat> crop_img_;
    int curr_feature_map_w_;
    int curr_frature_map_h_;
    int num_channels_;
    int minSize;
    double* threshold;
    double factor;
    bool use_gpu;
};

#endif //SRC_TRAFFIC_SIGN_H
