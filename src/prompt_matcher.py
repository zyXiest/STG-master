def match_prompt(question_content, templ_values):

    question_list_01 = ['Is this sound from the instrument in the video?']
    question_list_02 = ['Is the <Object> in the video always playing?']
    question_list_03 = ['Is there a voiceover?']
    question_list_04 = ['How many instruments are sounding in the video?']
    question_list_05 = ['How many types of musical instruments sound in the video?']
    question_list_06 = ['How many instruments in the video did not sound from beginning to end?']
    question_list_07 = ['How many sounding <Object> in the video?']
    question_list_08 = ['Where is the <LL> instrument?']
    question_list_09 = ['Is the <FL> sound coming from the <LR> instrument?']
    question_list_10 = ['Which is the musical instrument that sounds at the same time as the <Object>?']
    question_list_11 = ['What is the <LR> instrument of the <FL> sounding instrument?']
    question_list_12 = ['Is the instrument on the <LR> more rhythmic than the instrument on the <LR>?']
    question_list_13 = ['Is the instrument on the <LR> louder than the instrument on the <LR>?']
    question_list_14 = ['Is the <Object> on the <LR> more rhythmic than the <Object> on the <LR>?']
    question_list_15 = ['Is the <Object> on the <LR> louder than the <Object> on the <LR>?']
    question_list_16 = ['Where is the <FL> sounding instrument?']
    question_list_17 = ['Which <Object> makes the sound <FL>?']
    question_list_18 = ['What is the <TH> instrument that comes in?']
    question_list_19 = ['Which instrument makes sounds <BA> the <Object>?']
    question_list_20 = ['Is there a <Object> in the entire video?']
    question_list_21 = ['Are there <Object> and <Object> instruments in the video?']
    question_list_22 = ['How many types of musical instruments appeared in the entire video?']
    question_list_23 = ['How many <Object> are in the entire video?']
    question_list_24 = ['Where is the performance?']
    question_list_25 = ['What is the instrument on the <LR> of <Object>?']
    question_list_26 = ['What kind of musical instrument is it?']
    question_list_27 = ['What kind of instrument is the <LRer> instrument?']
    question_list_28 = ['Is there a <Object> sound?']
    question_list_29 = ['Are there <Object> and <Object> sound?']
    question_list_30 = ['How many musical instruments were heard throughout the video?']
    question_list_31 = ['Is the <Object> more rhythmic than the <Object>?']
    question_list_32 = ['Is the <Object> louder than the <Object>?']
    question_list_33 = ['Is the <Object> playing longer than the <Object>?']

    qa_prompt = []
    if question_content in question_list_01:
        qa_prompt_01 = "The sound is from the instrument in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_02:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_02 = "The " + instrument + " is not playing in this video."
        qa_prompt.append(qa_prompt_02)
    elif question_content in question_list_03:
        qa_prompt_02 = "There are sounds other than musical instruments in the video."
        qa_prompt.append(qa_prompt_02)
    elif question_content in question_list_04:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_05:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_06:
        qa_prompt_01 = "The instrument is not playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_07:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "The " + instrument + " is playing in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_08:
        qa_prompt_01 = "The sounds of musical instruments in the video are different."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_09:
        qa_prompt_01 = "The instruments in the video are not sounding simultaneously."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_10:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "The " + instrument + " is playing in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_11:
        first_last = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        left_right = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "There are musical instruments on the " + left_right + " that are not being played."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_12:
        qa_prompt_01 = "Inconsistent rhythmic sense of instrumental performance in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_13:
        qa_prompt_01 = "The sounds of musical instruments in the video are different."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_14:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        left_right_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[1]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[2]
        left_right_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument_01 + " on the " + left_right_01 + " plays a different rhythm than the " + instrument_02 + " on the " + left_right_02 + "."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_15:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        left_right_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[1]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[2]
        left_right_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument_01 + " on the " + left_right_01 + " and the " + instrument_02 + " on the " + left_right_02 + " produce different volumes of sound."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_16:
        qa_prompt_01 = "The instruments in the video do not sound simultaneously."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_17:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "The " + instrument + " in the video are not sounding at the same time."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_18:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_19:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument + " is playing in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_20:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "The " + instrument + " is not in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_21:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "There are instruments other than " + instrument_01 + " or " + instrument_02 + " in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_22:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_23:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "The " + instrument + " is in this video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_24:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_25:
        left_right = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "There is a musical instrument on the " + left_right + " side of the " + instrument + "."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_26:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_27:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_28:
        instrument = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        qa_prompt_01 = "There are sounds of instruments other than the " + instrument + " in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_29:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "There are sounds of instruments other than the " + instrument_01 + " or the " + instrument_02 + " in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_30:
        qa_prompt_01 = "There are musical instruments playing in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_31:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument_01 + " and " + instrument_02 + " have different rhythms in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_32:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument_01 + " and " + instrument_02 + " have different sounds in the video."
        qa_prompt.append(qa_prompt_01)
    elif question_content in question_list_33:
        instrument_01 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[0]
        instrument_02 = templ_values.replace("\"", "").replace("[", "").replace("]", "").replace(" ","").split(",")[-1]
        qa_prompt_01 = "The " + instrument_01 + " and " + instrument_02 + " are not played at the same time in the video."
        qa_prompt.append(qa_prompt_01)
    else:
        qa_prompt = 'error!'

    return qa_prompt[0]