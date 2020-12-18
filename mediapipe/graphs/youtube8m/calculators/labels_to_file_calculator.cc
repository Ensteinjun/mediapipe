#include <math.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <fstream>


// for convenience
#include "mediapipe/graphs/youtube8m/calculators/nlohmann_json.hpp"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {
//
// Usage example:
// node {
//   calculator: "LabelsToFileCalculator"
//   input_stream: "LABELS:top_k_labels"
//   input_stream: "SCORES:top_k_scores"
//   input_side_packet: "OUTPUT_LABEL_PATH:output_label_path"
// }

class LabelsToFileCalculator: public CalculatorBase {
    public:
        static mediapipe::Status GetContract(CalculatorContract* cc);
        mediapipe::Status Open(CalculatorContext* cc) override;
        mediapipe::Status Process(CalculatorContext* cc) override;
        mediapipe::Status Close(CalculatorContext* cc) override {
            LOG(INFO) << "Save Labels To File: " << _output_label_path;
            std::ofstream label_file_stream(_output_label_path);
            label_file_stream << _labels_data.dump(2) << std::endl;
            return mediapipe::OkStatus();
        }
    private:
        std::string _output_label_path;
        nlohmann::json _labels_data;
};

REGISTER_CALCULATOR(LabelsToFileCalculator);

mediapipe::Status LabelsToFileCalculator::GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("LABELS").Set<std::vector<std::string>>();
    cc->Inputs().Tag("SCORES").Set<std::vector<float>>();
    cc->InputSidePackets().Tag("OUTPUT_LABEL_PATH").Set<std::string>();
    return mediapipe::OkStatus();
}

mediapipe::Status LabelsToFileCalculator::Open(CalculatorContext* cc) {
    _labels_data = nlohmann::json::array();
    _output_label_path = cc->InputSidePackets().Tag("OUTPUT_LABEL_PATH").Get<std::string>();
    LOG(INFO) << "OUTPUT_LABEL_PATH: " << _output_label_path;
    return mediapipe::OkStatus();
}

mediapipe::Status LabelsToFileCalculator::Process(CalculatorContext* cc) {
    nlohmann::json tmp_data = "{}"_json;
    tmp_data["labels"] = "[]"_json;
    tmp_data["timestamp"] = cc->InputTimestamp().Value();

    const std::vector<std::string>& label_vector = cc->Inputs().Tag("LABELS").Get<std::vector<std::string>>();
    const std::vector<float>& score_vector = cc->Inputs().Tag("SCORES").Get<std::vector<float>>();
    CHECK_EQ(label_vector.size(), score_vector.size());

    for (int i = 0; i < label_vector.size(); ++i) {
        nlohmann::json tmp_label = "{}"_json;
        tmp_label["name"] = label_vector[i];
        tmp_label["score"] = score_vector[i];
        tmp_data["labels"].push_back(tmp_label);
        LOG(INFO) << "File Timestamp: " << cc->InputTimestamp().Value() / 1000000.0 \
            << ", " << label_vector[i] << ":" << score_vector[i];
    }
    _labels_data.push_back(tmp_data);

    return mediapipe::OkStatus();
}

}  // namespace mediapipe
