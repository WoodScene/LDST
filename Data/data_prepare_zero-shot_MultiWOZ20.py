
# instruction + input + output

# zero-shot experiment



import random
import json
import sys
random.seed(42)

def Template1(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "

    if "Possible Values" in dial[1]:
        dial2 = dial[1].split(" [Possible Values] ")
        
        input1 = input1 + "[domain] " + dial2[0] + " "
        if random.random() > 0.5:
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
        
    else:
        input1 = input1 + "[domain] " + dial[1] + " "

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
        
        input1 = input1 + "[domain] " + dial2[0] + " "
        if random.random() > 0.5:
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
        
    else:
        input1 = input1 + "[domain] " + dial[1] + " "

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


if __name__ == '__main__':
    frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}
    # 2.0 ['hotel', 'train', 'attraction', 'restaurant', 'hospital', 'taxi', 'bus']
    # 从 train taxi hotel  restaurant attraction 中选择一个作为测试的domain

    except_domain = "attraction" # delete this domain from traning set
    activate_number = 471912
    
    except_number = 0
    original_number = 0
    for data_type in ["train"]:
        data_dir = "./MULTIWOZ20_preprocess/"+ data_type +".json"
        data_idx = "./MULTIWOZ20_preprocess/"+ data_type +".idx"
        
        output_filename = "./MULTIWOZ20_preprocess/"+ data_type +"_LLM_zero-shot_except-"+ str(except_domain) +"-domain.json"
        dataset_data = []
        
        idx_lines = open(data_idx).readlines()
        test_data_lines = open(data_dir).readlines()
        
        assert len(idx_lines) == len(test_data_lines)
        
        none_sample_number = len(idx_lines) - activate_number
        none_sample_select_ratio = round((activate_number/none_sample_number),3)
        print(f"从none的样本中，以{none_sample_select_ratio}的概率进行挑选，从而保证二者的数量均衡")
        #sys.exit(1)
        for idx_ in range(len(idx_lines)):
            dial_text = eval(test_data_lines[idx_].strip())
            #print(dial_text['dialogue'])
            if dial_text['state'] == "NONE":
                if random.random() > none_sample_select_ratio:
                    continue
            
            original_number += 1
            item = {}
            idx_list = idx_lines[idx_].strip()
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 

            if d_name == except_domain:
                except_number +=1 
                continue

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
            # if random.random() > 0.8:
            #     dataset_data.append(item)
            dataset_data.append(item)
            #print(item)
            #break
            
        
        with open(output_filename, 'w') as f:
            json.dump(dataset_data, f, indent=4)    
    print("done.")
    print(f"原本的样本数量为：{original_number}。被删掉的样本数量为：{except_number}。最终该训练集的样本总数为：{len(dataset_data)}")