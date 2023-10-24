

import random
import json
import sys

def Template1(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "

    if "Possible Values" in dial[1]:
        dial2 = dial[1].split(" [Possible Values] ")
        
        input1 = input1 + "[domain] " + dial2[0] + ". "
        if random.random() > 0.5:
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
        
    else:
        input1 = input1 + "[domain] " + dial[1] + ". "

    input1 = input1 + "If the slot is not mentioned in the dialogue, just return NONE. \n "
    
    if random.random() > 0.5:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is \n"
    else:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "
    
    output1 = dial_text['state']
    return input1, output1



def Template2(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "

    if "Possible Values" in dial[1]:
        dial2 = dial[1].split(" [Possible Values] ")
        
        input1 = input1 + "[domain] " + dial2[0] + ". "
        if random.random() > 0.5:
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
        
    else:
        input1 = input1 + "[domain] " + dial[1] + ". "

    input1 = input1 + "\n "
    if random.random() > 0.5:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is \n"
    else:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "
    
    output1 = dial_text['state']
    return input1, output1
  

def Template3(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "
    if random.random() > 0.5:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is \n"
    else:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "

    output1 = dial_text['state']
    return input1, output1

def Count_activate_state_number():
    data_dir = "./SGD_preprocess/"+ "train" +".json"
    test_data_lines = open(data_dir).readlines()
    activate_number = 0
    for idx_ in range(len(test_data_lines)):
        dial_text = eval(test_data_lines[idx_].strip())
        if dial_text['state'] != "NONE":
            activate_number += 1
    return activate_number

if __name__ == '__main__':
    # 1 5 10 25
    few_shot_ratio = 1
    #activate_number = Count_activate_state_number() # 非none state的样本数
    activate_number = 86367
    print(f"非none state的样本数为{activate_number}")
    
    for data_type in ["train"]:
        data_dir = "./SGD_preprocess/"+ data_type +".json"
        data_idx = "./SGD_preprocess/"+ data_type +".idx"
        
        output_filename = "./SGD_preprocess/"+ data_type +"_LLM_few-shot-"+ str(few_shot_ratio) +"percent.json"
        dataset_data = []
        
        idx_lines = open(data_idx).readlines()
        test_data_lines = open(data_dir).readlines()
        
        assert len(idx_lines) == len(test_data_lines)
        
        total_select_number = int(len(idx_lines)*few_shot_ratio/100)
        print(f"样本总数是{len(idx_lines)}，需要从中选{total_select_number}个样本")
        
        
        select_none_number = int(total_select_number/2)
        select_activate_number = total_select_number -select_none_number
        if select_activate_number > activate_number:
            select_activate_number = activate_number
            select_none_number = total_select_number - activate_number
        print(f"其中是none的样本选{select_none_number}个，非none的样本选{select_activate_number}个")
        
        select_none_ratio = round(select_none_number/(len(idx_lines) - activate_number),5)
        select_activate_ratio = round(select_activate_number/activate_number,5)
        print(f"所以，需要以{select_none_ratio}的概率从none的state中挑选样本，以{select_activate_ratio}的概率从非none的state中挑选样本")
        #sys.exit(1)
        
        for idx_ in range(len(idx_lines)):
            dial_text = eval(test_data_lines[idx_].strip())
            #print(dial_text['dialogue'])
            if dial_text['state'] == "NONE":
                if random.random() > select_none_ratio:
                    continue
            else:
                if random.random() > select_activate_ratio:
                    continue
            
            item = {}
            idx_list = idx_lines[idx_].strip()
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 
            if random.random() > 0.1:
                instru = "Track the state of the slot in the input dialogue."
            else:
                instru = "Track the state of the slot <"+ d_name+ "-" + s_name +"> in the input dialogue."
    
            # 然后随机选择template 
            template_id = random.randint(1, 3)
            #template_id =3
            if template_id == 1:
                input1, output1 = Template1(dial_text)
            elif template_id == 2:
                input1, output1 = Template2(dial_text)
            elif template_id == 3:
                input1, output1 = Template3(dial_text)
            else:
                print("error")
                sys.exit(1)
            
            item['instruction'] = instru
            item['input'] = input1
            item['output'] = output1
            dataset_data.append(item)
            #print(item)
            #break
            
        
        with open(output_filename, 'w') as f:
            json.dump(dataset_data, f, indent=4)    
    print("done.")
    print(f"最终该训练集的样本总数为：{len(dataset_data)}")