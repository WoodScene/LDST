
import os
import sys
import json
from glob import glob
from random import sample

domain_desc_flag = True # To append domain descriptions or not 
slot_desc_flag = True  # To append slot descriptions or not 
PVs_flag = True # for categorical slots, append possible values as suffix

def preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split, slot_dec):
  dial_json_n = dial_json.split("/")[-1]
  dial_json = open(dial_json)
  dial_json = json.load(dial_json)
#   print(len(dial_json))# 10438
#   sys.exit(1)
  count = 0
  for dial_idx in dial_json.keys():
    #print(dial_idx)

    if split == "train":
        if dial_idx in test_json_list or dial_idx in val_json_list:
            continue
    elif split == "dev":
        if dial_idx not in val_json_list :
            continue
    elif split == "test":
        if dial_idx not in test_json_list :
            continue
    else:
        print("error")
        sys.exit(1)

    dial = dial_json[dial_idx]['log']
    #print(dial)
    #sys.exit(1)
    cur_dial = ""

    for turn_id in range(len(dial)):
      #print("turn id :", turn_id)
      content = dial[turn_id]
      #print(len(content['metadata']))
      if len(content['metadata']) == 0:
        speaker = " [USER] " 
        #assert turn_id %2 == 0
      else:
        speaker = " [SYSTEM] "   
        #assert turn_id %2 == 1
      uttr = content['text']
      cur_dial += speaker
      cur_dial += uttr  

      if len(content['metadata']) > 0:
        active_slot_values = {}
        turn_slot_dic = content['metadata']
        for frame_key in turn_slot_dic.keys():
          #print(frame_key) #taxi
          if frame_key not in domain_list:
              continue
          #print(turn_slot_dic[frame_key]['semi'])
          for slot_act in turn_slot_dic[frame_key]['semi'].keys():
              if turn_slot_dic[frame_key]['semi'][slot_act] != "":
                active_slot_values[frame_key.lower()+"-"+slot_act.lower()] = turn_slot_dic[frame_key]['semi'][slot_act] # 这也不一样
          for slot_act in turn_slot_dic[frame_key]['book'].keys():
              if slot_act != "booked" and turn_slot_dic[frame_key]['book'][slot_act] != "":
                active_slot_values[frame_key.lower()+"-book"+slot_act.lower()] = turn_slot_dic[frame_key]['book'][slot_act] # 这也不一样
        # if len(active_slot_values) > 0:
        #print("active_slot_values: ", active_slot_values) #{'hotel-name': 'not mentioned', 'hotel-area': 'not mentioned', 'hotel-parking': 'not mentioned', 'hotel-pricerange': 'cheap', 'hotel-stars': 'not mentioned', 'hotel-internet': 'not mentioned', 'hotel-type': 'hotel'}
        #sys.exit(1)
        # iterate thourgh each domain-slot pair in each user turn 
        for domain in schema.keys():
          # skip domains that are not in the testing set

          slots = [domain.split("-")[1]]
          d_name = domain.split("-")[0]
          #print(d_name, slots)
          for slot in slots:
            s_name = slot

            #print(d_name,s_name)
            # new 匹配一下des
            slot_description = ""
            count_des = 0
            for des_key in slot_dec.keys():
              slot_t = des_key.split("-")[1]
              while " " in slot_t:
                slot_t = slot_t.replace(" ", "")
              slot_t = slot_t.lower()
              slot_t2 = s_name.lower()
              while " " in slot_t2:
                slot_t2 = slot_t2.replace(" ", "")
              if d_name in des_key and slot_t == slot_t2:
                slot_description = slot_dec[des_key][0]
                count_des += 1

            if slot_description == "" or count_des != 1:
              print("error des", count_des)
              sys.exit(1)
            #print("slot_description: ", slot_description)
            # generate schema prompt w/ or w/o natural langauge descriptions
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + ","
            #schema_prompt += " [slot] " + s_name + "."
            schema_prompt += " [slot] " + s_name + ", it indicates " + slot_description if slot_desc_flag  else s_name
            
            if PVs_flag:
              # only append possible values if the slot is categorical
              if len(schema[domain]) > 0:
                PVs = ", ".join(schema[domain])
                #print(PVs)
                #sys.exit(1)
                schema_prompt += " [Possible Values] " + PVs
            #print("schema prompt:", schema_prompt)
            #sys.exit(1)
            domain_slot = d_name.lower() + "-" + s_name.lower()
            s_name2 = s_name
            while " " in s_name2:
                s_name2 = s_name2.replace(" ", "")
            domain_slot2 = d_name.lower() + "-" + s_name2.lower()
            if domain_slot in active_slot_values.keys():
              target_value = active_slot_values[domain_slot]
              # print(target_value)
              # sys.exit()
            elif domain_slot2 in active_slot_values.keys():
              target_value = active_slot_values[domain_slot2]
              # print("laizhele ")
              # print(domain_slot2)
              # sys.exit(1)
            else:
              # special token for non-active slots
              target_value = "NONE"
            
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }
            
            # print("line: ", line)
            # sys.exit()
            #print()
            #print()
            out.write(json.dumps(line))
            out.write("\n")
            #count +=1
            #if slot["name"] in active_slot_values.keys():
            #  sys.exit(1)
            # write idx file for post-processing deocding
            idx_list = [ dial_json_n, str(dial_idx), str(turn_id), str(frame_idxs[d_name]), d_name, s_name ]
            #print(idx_list)
            #sys.exit(1)
            idx_out.write("|||".join(idx_list))
            idx_out.write("\n")
      count +=1

      #sys.exit(1)
      #if count >2:
      #  sys.exit(1)
  return


def Generate_data(domain_list):
    data_path = "./MULTIWOZ2.4/"
    data_path21 = "./MultiWOZ_2.1/"
    #data_path = sys.argv[1]

    parent_dir = os.path.dirname(os.path.dirname(data_path))
    multiwoz_dir = os.path.basename(os.path.dirname(data_path))
    data_path_out = os.path.join(parent_dir, multiwoz_dir+"_preprocess")
    if not os.path.exists(data_path_out):
      os.makedirs(data_path_out)
    print(data_path_out)
    #sys.exit(1)

    frame_idxs = {}
    for service_idx in range(len(domain_list)):
        service = domain_list[service_idx]
        frame_idxs[service] = service_idx
    #print(frame_idxs) #{'hotel': 0, 'train': 1, 'attraction': 2, 'restaurant': 3, 'hospital': 4, 'taxi': 5, 'bus': 6}
    #sys.exit(1)
    #frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}

    # skip domains that are not in the testing set
    #excluded_domains = ["police", "hospital", "bus"]
    
    test_list_file = open(os.path.join(data_path, "{}.json".format("testListFile")))
    test_json_list = []
    for lin in test_list_file:
        test_json_list.append(lin.strip())
    
    val_list_file = open(os.path.join(data_path, "{}.json".format("valListFile")))
    val_json_list = []
    for lin in val_list_file:
        val_json_list.append(lin.strip())
    # print(val_json_list)
    # sys.exit()
    
    for split in ["train","dev", "test"]:
    #for split in [ "test"]:
        print("--------Preprocessing {} set---------".format(split))
        out = open(os.path.join(data_path_out, "{}.json".format(split)), "w") # W 模式 会覆盖之前的内容
        idx_out = open(os.path.join(data_path_out, "{}.idx".format(split)), "w")
        dial_jsons = glob(os.path.join(data_path, "*json"))
        
        schema_path = data_path + "ontology.json"
        schema = json.load(open(schema_path))
        
        slot_path = data_path21 + "slot_descriptions.json"
        slot_dec = json.load(open(slot_path))
        #print(slot_dec)
        #sys.exit(1)

        for dial_json in dial_jsons:
            if "data.json" in dial_json:
                preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split, slot_dec)
        idx_out.close()
        out.close()
    print("--------Finish Preprocessing---------")



def Analysis():
    data_path = "./MULTIWOZ2.4/"

    schema_path = data_path + "ontology.json"
    schema = json.load(open(schema_path))


    domain_list = []
    slot_list = []

    for domain in schema.keys():
        #print(domain['service_name'])
        d = domain.split("-")[0]
        slot_list.append(domain)
        if d not in domain_list:
            domain_list.append(d)
   
    print(domain_list)
       
    print(slot_list)

    print(len(domain_list))
    print(len(slot_list))


    return domain_list

def main():
    domain_list = Analysis()
    print(domain_list)
    Generate_data(domain_list)

if __name__=='__main__':
    main()
