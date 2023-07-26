#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

//variables for Canny edge detection
cv::Mat canny_source, canny_source_gray;
cv::Mat canny_result_img, canny_detected_edges;
int canny_low_thresh = 40;
const int canny_max_low_tresh = 100;
int canny_ratio = 3;
const int canny_kernel_size = 3;
const char* canny_window_name = "Edge Map";

static void CannyThreshold(int, void*){
    blur(canny_source_gray, canny_detected_edges, Size(3,3));
    Canny(canny_detected_edges, canny_detected_edges, canny_low_thresh, canny_low_thresh*canny_ratio, canny_kernel_size);
    canny_result_img = Scalar::all(0);
    canny_source.copyTo(canny_result_img, canny_detected_edges);
    imshow(canny_window_name, canny_result_img);
}

class myCropLayer : public Layer{
public:
    myCropLayer(const LayerParams &params) : Layer(params){
    }
    static cv::Ptr<Layer> create(LayerParams& params){
        return cv::Ptr<Layer>(new myCropLayer(params));
    }
    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE{
        CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = inputs[1][2];
        outShape[3] = inputs[1][3];
        outputs.assign(1, outShape);
        return false;
    }
    virtual void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals) CV_OVERRIDE{
        cv::Mat * inp = input[0];
        cv::Mat  out = output[0];
        int ystart = (inp->size[2] - out.size[2]) / 2;
        int xstart = (inp->size[3] - out.size[3]) / 2;
        int yend = ystart + out.size[2];
        int xend = xstart + out.size[3];

        const int batchSize = inp->size[0];
        const int numChannels = inp->size[1];
        const int height = out.size[2];
        const int width = out.size[3];

        int sz[] = { (int)batchSize, numChannels, height, width };
        out.create(4, sz, CV_32F);
        for(int i=0; i<batchSize; i++){
            for(int j=0; j<numChannels; j++){
                cv::Mat plane(inp->size[2], inp->size[3], CV_32F, inp->ptr<float>(i,j));
                cv::Mat crop = plane(cv::Range(ystart,yend), cv::Range(xstart,xend));
                cv::Mat targ(height, width, CV_32F, out.ptr<float>(i,j));
                crop.copyTo(targ);
            }
        }
    }
};


int main( int argc, char* argv[] ){
    std::string img_name = argv[1];

    // loading the model and image
    CV_DNN_REGISTER_LAYER_CLASS(Crop, myCropLayer);
    Net net = readNet("deploy.prototxt", "hed_pretrained_bsds.caffemodel");
    cv::Mat img = cv::imread("../input_images/" + img_name + ".png");

    // preparing the input to network
    cv::Mat HED_input;
    resize(img, HED_input, img.size());
    cv::Mat blob = blobFromImage(HED_input, 0.5, img.size(), cv::Scalar(cv::mean(img)[0],cv::mean(img)[1],cv::mean(img)[2]), false, false);
    
    // providing input and gettin the output
    net.setInput(blob);
    cv::Mat HED_output = net.forward(); 

    // converting tensor back to image
    std::vector<cv::Mat> vectorOfImagesFromBlob;
    imagesFromBlob(HED_output, vectorOfImagesFromBlob);
    cv::Mat tmpMat = vectorOfImagesFromBlob[0] * 255;
    cv::Mat tmpMatUchar;
    tmpMat.convertTo(tmpMatUchar, CV_8U);

    // saving the result
    cv::resize(tmpMatUchar, HED_output, img.size());
    cv::imwrite("../result_images/" + img_name + "_HED_f05.png", HED_output);
   
    // blurring the HED output and performing segmentation
    cv::Mat blurred_HED_img;
    cv::GaussianBlur(HED_output, blurred_HED_img, cv::Size(3,3), 0);
    cv::Mat thresholded_HED_img;
    cv::threshold(blurred_HED_img, thresholded_HED_img, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    cv::imwrite("../result_images/segmentation_thresh_" + img_name + ".png", thresholded_HED_img);

    // Performing connected component labeling
    cv::Mat labels, stats, centroids;
    int n_labels = connectedComponentsWithStats(thresholded_HED_img, labels, stats, centroids, 4);

    std::vector<cv::Vec3b> colors(n_labels);
    for(int i = 0; i < n_labels; i++)
        colors[i] = cv::Vec3b(rand()&255,rand()&255,rand()&255);
    cv::Mat colour_segmented_HED_img = cv::Mat::zeros(thresholded_HED_img.size(),CV_8UC3);

    for(int i = 0; i < thresholded_HED_img.cols; i++){
        for(int j = 0; j < thresholded_HED_img.rows; j++){
            if(labels.at<int>(cv::Point(i,j)) != 0){
                colour_segmented_HED_img.at<cv::Vec3b>(cv::Point(i,j)) = colors[(int)labels.at<int>(cv::Point(i,j))];
            }
        }
    }
    
    cv::waitKey();
    cv::imwrite("../result_images/colour_segmented_" + img_name + ".png", colour_segmented_HED_img);
    
    //Canny edge detection
    canny_source = cv::imread("../input_images/" + img_name + ".png");
    cv::cvtColor(canny_source, canny_source_gray, COLOR_BGR2GRAY);
    CannyThreshold(0, 0);

    cv::imwrite("../result_images/" + img_name + "_canny_th_40.png", canny_result_img);
    return 0;
}