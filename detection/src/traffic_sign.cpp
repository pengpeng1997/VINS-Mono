//
// Created by pengpeng on 2019/10/17.
//

#include "traffic_sign.h"
#include<typeinfo>

CascadeCNN::CascadeCNN(const std::string &model_dir, const bool use_gpu):use_gpu(use_gpu) {
    std::string pnet_filename = model_dir + "pnet_epoch_10_traced.pt";
    PNet_ = torch::jit::load(pnet_filename);
    std::cout << "Import PNet successfully" << std::endl;

    std::string rnet_filename = model_dir + "rnet_epoch_10_traced.pt";
    RNet_ = torch::jit::load(rnet_filename);
    std::cout << "Import RNet successfully" << std::endl;

}

void CascadeCNN::img2tensor(cv::Mat & img, at::Tensor & tensor)
{
    if(use_gpu) tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte).to(torch::kCUDA);
    else tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte).to(torch::kCPU);
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.toType(torch::kFloat);
    tensor = tensor.div(255);
}

// compare score
bool CompareBBox(const ObjInfo & a, const ObjInfo & b) {
    return a.bbox.score > b.bbox.score;
}

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<ObjInfo> CascadeCNN::NonMaximumSuppression(std::vector<ObjInfo>& bboxes,
                                                       float thresh,char methodType){
    std::vector<ObjInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        ObjRect select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.x2-select_bbox.x1+1) * (select_bbox.y2-select_bbox.y1+1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            ObjRect& bbox_i = bboxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2-bbox_i.x1+1) * (bbox_i.y2-bbox_i.y1+1));
            float area_intersect = w * h;

            switch (methodType) {
                case 'u':
                    if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                        mask_merged[i] = 1;
                    break;
                case 'm':
                    if (static_cast<float>(area_intersect) / std::min(area1 , area2) > thresh)
                        mask_merged[i] = 1;
                    break;
                default:
                    break;
            }
        }
    }
    return bboxes_nms;
}

void CascadeCNN::Bbox2Square(std::vector<ObjInfo>& bboxes){
    for(int i = 0; i < bboxes.size(); i++){
        float w = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
        float h = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
        float side = w > h ? w : h;
        bboxes[i].bbox.x1 += (w - side) * 0.5;
        bboxes[i].bbox.y1 += (h - side) * 0.5;

        bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
        bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
        bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
        bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

    }
}

std::vector<ObjInfo> CascadeCNN::BoxRegress(std::vector<ObjInfo>& objInfo,int stage){//stage??
    std::vector<ObjInfo> bboxes;
    for(int bboxId = 0; bboxId < objInfo.size();bboxId++){
        ObjRect objRect;
        ObjInfo tempObjInfo;
        float regw = objInfo[bboxId].bbox.x2 - objInfo[bboxId].bbox.x1;
        regw += (stage == 1)? 0:1;
        float regh = objInfo[bboxId].bbox.y2 - objInfo[bboxId].bbox.y1;
        regh += (stage == 1)? 0:1;
        objRect.x1 = objInfo[bboxId].bbox.x1 + regw * objInfo[bboxId].regression[0];
        objRect.y1 = objInfo[bboxId].bbox.y1 + regh * objInfo[bboxId].regression[1];
        objRect.x2 = objInfo[bboxId].bbox.x2 + regw * objInfo[bboxId].regression[2];
        objRect.y2 = objInfo[bboxId].bbox.y2 + regh * objInfo[bboxId].regression[3];
        objRect.score = objInfo[bboxId].bbox.score;
        tempObjInfo.bbox = objRect;
        tempObjInfo.regression = objInfo[bboxId].regression;
        bboxes.push_back(tempObjInfo);
//        std::cout<<"before:"<<objInfo[bboxId].bbox.x1<<","<<objInfo[bboxId].bbox.y1<<","<<objInfo[bboxId].bbox.x2<<","<<objInfo[bboxId].bbox.y2<<std::endl;
//        std::cout<<"reg and w h:"<<regw<<","<<regh<<","<<objInfo[bboxId].regression<<std::endl;
//        std::cout<<"after:"<<objRect.x1<<","<<objRect.y1<<","<<objRect.x2<<","<<objRect.y2<<std::endl;
    }
//    return bboxes;
    return objInfo;
}

// compute the padding coordinates (pad the bounding boxes to square)
void CascadeCNN::Padding(int img_w,int img_h){
    for(int i = 0; i < regressed_rects_.size(); i++){
        ObjInfo tempObjInfo;
        tempObjInfo = regressed_rects_[i];
        tempObjInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_h) ? img_h : regressed_rects_[i].bbox.y2;
        tempObjInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_w) ? img_w : regressed_rects_[i].bbox.x2;
        tempObjInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
        tempObjInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
        regressed_padding_.push_back(tempObjInfo);
    }
}

void CascadeCNN::GenerateBoundingBox(at::Tensor cls, at::Tensor reg, float scale, float threshold, const int ws, const int hs) {
//candidate_rects_;
    int stride = 2;
    int cellSize = 12;

    int curr_feature_map_w_ = std::floor((ws - cellSize) * 1.0 / stride) + 1;
    int curr_feature_map_h_ = std::floor((hs - cellSize) * 1.0 / stride) + 1;

    int regOffset = curr_feature_map_w_ * curr_feature_map_h_;

    std::cout<<"ws, hs, scale:"<<ws<<","<<hs<<","<<scale<<std::endl;
    std::cout<<" cls.sizes():" <<  cls.sizes() <<std::endl;
    std::cout<<" reg.sizes():" <<  reg.sizes() <<std::endl;
//    std::cout<<" cls[0][0][0][0]:"<< cls[0][0][0][0] <<std::endl;
    std::cout<<"map_w, h:"<<curr_feature_map_w_<<","<<curr_feature_map_h_<<std::endl;
//    for (int i = 0; i < 10; ++i) {
//        std::cout << "score: " << top_scores[i].item().toFloat() << "\t" << "label: " << labels[top_idxs[i].item().toInt()] << std::endl;
//    }
    for(int i = 0; i < curr_feature_map_w_; i++) {
        for(int j = 0; j < curr_feature_map_h_; j++) {
            float prob = cls[0][0][j][i].item().toFloat();
            if(prob > threshold) {
                ObjRect rect;
                rect.x1 = stride * i / scale;
                rect.y1 = stride * j / scale;
                rect.x2 = (stride * i + cellSize) / scale;
                rect.y2 = (stride * j + cellSize) / scale;
                rect.score = prob;
                cv::Vec4f regression(reg[0][0][j][i].item().toFloat(), reg[0][1][j][i].item().toFloat(), reg[0][2][j][i].item().toFloat(), reg[0][3][j][i].item().toFloat());
                ObjInfo info;
                info.bbox = rect;
                info.regression = regression;
                candidate_rects_.push_back(info);
            }
        }
    }
}


void CascadeCNN::ClassifyObj_batch(const std::vector<ObjInfo>& regressed_rects,cv::Mat &sample_single,
                                   torch::jit::script::Module& net,double thresh,char netName) {
    int numBox = regressed_rects.size();
    int input_width = 24;
    int input_height = 24;
    candidate_rects_.clear();
    at::Tensor inputs;
    for(int i = 0; i < numBox; i++) {
        int pad_top   = std::abs(regressed_padding_[i].bbox.x1 - regressed_rects[i].bbox.x1);
        int pad_left  = std::abs(regressed_padding_[i].bbox.y1 - regressed_rects[i].bbox.y1);
        int pad_right = std::abs(regressed_padding_[i].bbox.y2 - regressed_rects[i].bbox.y2);
        int pad_bottom= std::abs(regressed_padding_[i].bbox.x2 - regressed_rects[i].bbox.x2);

        cv::Mat crop_img;
//        std::cout<<"regressed_padding_[i]:"<<regressed_padding_[i].bbox.y1<<","<<regressed_padding_[i].bbox.y2<<","<<regressed_padding_[i].bbox.x1<<","<<regressed_padding_[i].bbox.x2<<std::endl;
        crop_img = sample_single(cv::Range(regressed_padding_[i].bbox.y1-1, regressed_padding_[i].bbox.y2), cv::Range(regressed_padding_[i].bbox.x1-1, regressed_padding_[i].bbox.x2));

        cv::copyMakeBorder(crop_img, crop_img, pad_left, pad_right, pad_top, pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(crop_img, crop_img, cv::Size(input_width, input_height),0,0,cv::INTER_NEAREST);

//        crop_img.convertTo(crop_img, CV_32FC3);
        at::Tensor tensor;
        img2tensor(crop_img, tensor);
        if(i == 0) inputs = tensor;
        else inputs = torch::cat({inputs, tensor}, 0);
    }
    c10::intrusive_ptr<c10::ivalue::Tuple> output = net.forward({inputs}).toTuple();

    at::Tensor cls = output->elements()[0].toTensor();
    at::Tensor reg = output->elements()[1].toTensor();

//    std::cout<<"cls size:"<<cls.sizes()/*<<","<<cls[0][0]*/<<std::endl;
//    std::cout<<"reg size:"<<reg.sizes()<<std::endl;
    for(int i = 0; i < numBox; i++) {
        float prob = cls[i][0].item().toFloat();
        if(prob > thresh) {
            ObjRect rect;
            rect.x1 = regressed_rects[i].bbox.x1;
            rect.y1 = regressed_rects[i].bbox.y1;
            rect.x2 = regressed_rects[i].bbox.x2;
            rect.y2 = regressed_rects[i].bbox.y2;
            rect.score = prob;
            cv::Vec4f regression(reg[i][0].item().toFloat(), reg[i][1].item().toFloat(), reg[i][2].item().toFloat(), reg[i][3].item().toFloat());
            ObjInfo info;
            info.bbox = rect;
            info.regression = regression;
            candidate_rects_.push_back(info);
            std::cout<<"prob, rect, reg:"<<rect.score<<","<<rect.x1<<","<<rect.y1<<","<<rect.x2<<","<<rect.y2<<","<<regression<<std::endl;
        }

    }
    std::cout<<"final result size:"<<candidate_rects_.size()<<std::endl;

}


void CascadeCNN::Detect(const cv::Mat& img, std::vector<ObjInfo>& ObjInfo, int minSize, double* threshold, double factor) {
    this->minSize = minSize;
    this->factor = factor;

    cv::Mat image_clone, sample_single, resized;
    image_clone = img.clone();
    sample_single = img.clone();
//    cv::imshow("hh", sample_single);
//    cv::waitKey(1000);
//    cv::cvtColor(sample_single, sample_single, cv::COLOR_BGR2RGB);
//    sample_single.convertTo(sample_single, CV_32FC3);

//    sample_single = sample_single.t();

    int height = img.rows;
    int width = img.cols;
    int minWH = std::min(height, width);
//    num_channels_ = img.channels;

    int factor_count = 0;
    double m = 12./minSize;
    minWH *= m;
    std::vector<double> scales;
    while (minWH >= 12) {
        scales.push_back(m * std::pow(factor,factor_count));
        minWH *= factor;
        ++factor_count;
    }
    std::cout<<"factor_count:"<<factor_count<<std::endl;
    std::vector<at::Tensor> tensors;
    for(int i = 0; i < factor_count; i++) {
        double scale = scales[i];
        int ws = std::ceil(width * scale);
        int hs = std::ceil(height * scale);
        cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0);
//        cv::imshow("hh", resized);
//        cv::waitKey(1000);
//        resized.convertTo(resized, CV_32FC3);

        at::Tensor tensor;
        img2tensor(resized, tensor);
        std::cout<<"input size:"<<tensor.sizes()<<std::endl;

        c10::intrusive_ptr<c10::ivalue::Tuple> output = PNet_.forward({tensor}).toTuple();
        std::cout<<"forward successfully."<<std::endl;
        at::Tensor cls = output->elements()[0].toTensor();
        at::Tensor reg = output->elements()[1].toTensor();

        GenerateBoundingBox(cls, reg, scale, threshold[0], ws, hs);
        auto bboxes_nms = NonMaximumSuppression(candidate_rects_,0.5,'u');
        total_boxes_.insert(total_boxes_.end(),bboxes_nms.begin(),bboxes_nms.end());
    }

    int numBox = total_boxes_.size();
    if(numBox != 0){

        total_boxes_ = NonMaximumSuppression(total_boxes_,0.6,'u');
        regressed_rects_ = BoxRegress(total_boxes_,1);
        total_boxes_.clear();

        Bbox2Square(regressed_rects_);
        Padding(width,height);
//        plot(image_clone, regressed_rects_);

        /// Second stage
        ClassifyObj_batch(regressed_rects_,sample_single,RNet_,threshold[1],'r');//TODO
        plot(image_clone, candidate_rects_);
        candidate_rects_ = NonMaximumSuppression(candidate_rects_,0.7,'u');
        regressed_rects_ = BoxRegress(candidate_rects_,2);

        Bbox2Square(regressed_rects_);
        Padding(width,height);

        /// thired stage
        numBox = regressed_rects_.size();
        if(numBox != 0){
//            crop_img.convertTo(crop_img, CV_8UC3);
//            cv::imshow("hh", crop_img);
//            cv::waitKey(1000);
        }
    }
//    plot(image_clone, regressed_rects_);
    regressed_padding_.clear();
    regressed_rects_.clear();
    candidate_rects_.clear();
}

void CascadeCNN::plot(cv::Mat& img, std::vector<ObjInfo>& ObjInfo) {
    int thick = 2;
    CvScalar cyan = CV_RGB(0, 255, 255);
    ////    CvScalar blue = CV_RGB(0, 0, 255);
    for(auto info : ObjInfo) {
        cv::line(img, cv::Point(info.bbox.x1, info.bbox.y1), cv::Point(info.bbox.x1,info.bbox.y2), cyan, thick);
        cv::line(img, cv::Point(info.bbox.x1, info.bbox.y2), cv::Point(info.bbox.x2,info.bbox.y2), cyan, thick);
        cv::line(img, cv::Point(info.bbox.x2, info.bbox.y2), cv::Point(info.bbox.x2,info.bbox.y1), cyan, thick);
        cv::line(img, cv::Point(info.bbox.x2, info.bbox.y1), cv::Point(info.bbox.x1,info.bbox.y1), cyan, thick);
    }

    cv::imshow("hh", img);
    cv::waitKey();
}



