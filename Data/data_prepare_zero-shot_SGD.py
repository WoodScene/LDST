
import random
import json
import sys


if __name__ == '__main__':

    for data_type in ["test"]:
        data_dir = "./SGD_preprocess/"+ data_type +"_zero-shot.json"
        data_idx = "./SGD_preprocess/"+ data_type +"_zero-shot.idx"
        
        output_filename = "./SGD_preprocess/"+ data_type +"_LLM_zero-shot.json"
        dataset_data = []
        
        idx_lines = open(data_idx).readlines()
        test_data_lines = open(data_dir).readlines()
        
        assert len(idx_lines) == len(test_data_lines)
        
        
        for idx_ in range(len(idx_lines)):
            item = {}
            instru = "Track the state of the slot in the input dialogue."
            idx_list = idx_lines[idx_].strip()

            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 
            
            dial_text = eval(test_data_lines[idx_].strip())
            #print(dial_text['dialogue'])
            dial = dial_text['dialogue'].split(" [domain] ")
            input1 = ""
            input1 = input1 + dial[0] + "\n"
            flag = 0
            if "Possible Values" in dial[1]:
                dial2 = dial[1].split(" [Possible Values] ")
                
                input1 = input1 + "[domain] " + dial2[0] + ". "
                input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
                input1 = input1 + dial2[1] + ". "
                
            else:
                input1 = input1 + "[domain] " + dial[1] + ". "
                flag = 1
                
            #print(dial)
            input1 = input1 + "If the slot is not mentioned in the dialogue, just return NONE.\n"
            input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "
            # print(input1)
            # if flag == 1:
            #     sys.exit(1)
            
            output1 = dial_text['state']
            item['instruction'] = instru
            item['input'] = input1
            item['output'] = output1
            dataset_data.append(item)
        
        with open(output_filename, 'w') as f:
            json.dump(dataset_data, f, indent=4)    
            
    print("done.")
