import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from config import DEFAULT_OUTPUT_DIR


ast_to_concept = {
    "test" : "test",
    "treatment" : "treatment",
    "present" : "problem",
    "absent" : "problem",
    "possible" : "problem",
    "conditional" : "problem",
    "hypothetical" : "problem",
    "associated with someone else" : "problem"
}

token_id_to_label = {0: 'o',
                        1: 'test',
                        2: 'test',
                        3: 'treatment',
                        4: 'treatment',
                        5: 'present',
                        6: 'present',
                        7: 'absent',
                        8: 'absent',
                        9: 'possible',
                        10: 'possible',
                        11: 'conditional',
                        12: 'conditional',
                        13: 'hypothetical',
                        14: 'hypothetical',
                        15: 'associated with someone else',
                        16: 'associated with someone else'}


def save_predictions(test_data, predictions, ast_to_concept=ast_to_concept, token_id_to_label=token_id_to_label, output_dir=DEFAULT_OUTPUT_DIR):
    # see if ouput dir exists, if not create it
    if not os.path.exists(os.path.join(output_dir, "con")):
        os.makedirs(os.path.join(output_dir, "con"))
    if not os.path.exists(os.path.join(output_dir, "ast")):
        os.makedirs(os.path.join(output_dir, "ast"))     

    known_files = set({}) # set of files that have seen, because in the first time we clear the file
    for i,a in enumerate(test_data):
        
        #creat files
        con_output_dir = os.path.join(output_dir, "con")
        ast_output_dir = os.path.join(output_dir, "ast")

        open(os.path.join(con_output_dir, a['filename'] + ".con"), 'a').close()
        open(os.path.join(ast_output_dir, a['filename'] + ".ast"), 'a').close()
        
        
        
        if sum(predictions[i])==0:
            pass
        else:
            pred = predictions[i][0:len(a['input_ids'])-1] #remove padding zeros
            # -------------------------------- Real thing -------------------------------- #
            old_token = None
            splits = []
            for j,token in enumerate(pred):
                if old_token!=None and token_id_to_label[old_token] == token_id_to_label[token]: #continue same sequence
                    splits[-1][1] = j
                else: #start a new sequence
                    if len(splits) :
                        splits[-1][1] = j
                    old_token = token
                    splits.append([j, None, token])
            if splits[-1][1]==None:
                splits[-1][1] = splits[-1][0]+1
            for split in splits :
                token = split[-1]
                if token == 0:
                    continue
                mapping_list = a['offset_mapping'][split[0]:split[1]]
                mapping = [mapping_list[0][0], mapping_list[-1][1]] #by character
                

                

                # ------------------------------- word_mapping ------------------------------- #
                word = a["text"][mapping[0]:mapping[1]]
                word_mapping_0 = a["text"][:mapping[0]].count(" ")
                word_mapping_1 = a["text"][:mapping[1]].count(" ")
                word_mapping = [word_mapping_0, word_mapping_1]
                # word = " ".join(a['text'].split(" ")[word_mapping[0]:word_mapping[1]+1]).strip()
                
                
                # -------------------------------- Build lines ------------------------------- #
                con_line = 'c="'+ word
                con_line += '" ' 
                con_line += str(a["row"]) + ":" + str(word_mapping[0])
                con_line += ' '
                con_line += str(a["row"]) + ":" + str(word_mapping[1])
                con_line += '||t="'
                con_line += ast_to_concept[token_id_to_label[token]] + '"'
                if ast_to_concept[token_id_to_label[token]] == "problem":
                    ast_line = con_line
                    ast_line += '||a="'
                    ast_line += token_id_to_label[token]
                    ast_line += '"'
                
                # if a['filename'] == "0033":
                #     print("kj")
                
                # ---------------------------------------------------------------------------- #
                #                                    Concept                                   #
                # ---------------------------------------------------------------------------- #
                if a['filename'] + ".con" in known_files:
                    pass
                else : 
                    known_files.add(a['filename'] + ".con")
                    open(os.path.join(con_output_dir, a['filename'] + ".con"), 'w').close()
                with open(os.path.join(con_output_dir, a['filename'] + ".con"), 'a') as f:
                    f.write(con_line + "\n")
                    
                # ---------------------------------------------------------------------------- #
                #                                   Assertion                                  #
                # ---------------------------------------------------------------------------- #
                if ast_to_concept[token_id_to_label[token]] == "problem":
                    if a['filename'] + ".ast" in known_files:
                        pass
                    else : 
                        known_files.add(a['filename'] + ".ast")
                        open(os.path.join(ast_output_dir, a['filename'] + ".ast"), 'w').close()
                    with open(os.path.join(ast_output_dir, a['filename'] + ".ast"), 'a') as f:
                        f.write(ast_line + "\n")
                    