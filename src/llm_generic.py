
import base64
from concurrent.futures import ThreadPoolExecutor
import stuff
import numpy as np
import cv2
import json
import re
from typing import List, Dict

ALIAS_TO_FULL: Dict[str, str] = {
        "no_person":"no_person_visible","male":"is_male","female":"is_female",
        "hat":"is_wearing_hat_or_head_covering","mask":"is_wearing_a_mask_or_face_covering",
        "glasses":"wearing_glasses_or_sunglasses","beard":"has_facial_hair","long_hair":"has_shoulder_length_hair",
        "bald":"has_buzz_cut_or_bald_head","kid":"is_child","adult":"is_adult","senior":"is_senior","teen":"is_teen",
        "bag":"has_bag_or_backpack","uniform":"is_wearing_a_uniform","sleeves":"has_long_sleeves","tattoos":"has_visible_tattoos",
        "shorts":"is_wearing_shorts","bright":"is_wearing_bright_colored_clothing","coat":"is_wearing_a_coat_or_jacket",
        "weapon":"is_carrying_a_weapon","threatening":"has_a_threatening_posture","heavy_build":"has_heavy_build","lying":"is_lying_down",
        "top_light":"top_is_white_or_light","top_dark":"top_is_black_or_gray_or_dark","top_blue":"top_is_blue_or_purple",
        "top_green":"top_is_green","top_red":"top_is_red_or_pink","top_yellow":"top_is_orange_or_beige_or_yellow",
        "bottom_light":"bottom_is_white_or_light","bottom_dark":"bottom_is_black_or_gray_or_dark","bottom_blue":"bottom_is_blue_or_purple",
        "bottom_green":"bottom_is_green","bottom_red":"bottom_is_red_or_pink","bottom_yellow":"bottom_is_orange_or_beige_or_yellow",
        "smoking":"is_smoking_or_vaping",
        "logo":"clothing_has_prominent_logo",
        "hoodie":"is_wearing_hoodie",
        "phone":"is_holding_mobile_phone",
        "hi-vis":"is_wearing_hi_vis_clothes",
        "patterned":"has_patterned_clothing",
        "weapon_held":"weapon_held_in_hands",
        "dress":"is_wearing_a_dress_or_skirt",
        "running":"is_running",
        "fighting":"is_fighting",
        "behind_camera":"is_behind_camera",
    }

    # Precompile a regex that matches JSON object keys exactly (aliases) before a colon.
    # Pattern: "<alias>" (optionally with whitespace before colon) :
_ALIAS_KEY_RE = re.compile(
    r'(?<=")(' + "|".join(map(re.escape, ALIAS_TO_FULL.keys())) + r')(?="\s*:)'  # only keys, not values
)

class LLMGeneric:

    def _replace_alias_keys(self, json_like: str) -> str:
        return _ALIAS_KEY_RE.sub(lambda m: ALIAS_TO_FULL[m.group(1)], json_like)

    def expand_aliases_in_llm_output(self, raw: str) -> str:
        """
        Expand alias keys to full names by string replacement on the model's raw output.
        - Preserves ```json ... ``` fences if present.
        - Only replaces object KEYS that match aliases (safe: won't touch values).
        - Returns the expanded string (still minified/unchanged otherwise).
        """
        # find a ```json ... ``` block first; if none, any ```...```; else use whole text
        m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        fenced = True
        if not m:
            m = re.search(r"```(?:[a-zA-Z]+)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
            if not m:
                fenced = False

        if m:  # expand only the inner content, then rewrap
            inner = m.group(1)
            expanded = self._replace_alias_keys(inner)
            # reconstruct the original block as ```json ... ``` if it started that way, else keep same fence tag
            tag_m = re.search(r"^```([a-zA-Z]+)?", raw.strip())
            tag = tag_m.group(1) if tag_m and tag_m.group(1) else "json"
            return f"```{tag}\n{expanded}\n```"
        else:
            # no fences; just expand entire string
            return self._replace_alias_keys(raw)


    def process_images(self, s):
        r=self.client.infer(s["prompt"], images=[s["b64_image"]], attempts=2)
        #r=r.replace("null","false")
        #print(r)
        #print(r)
        r2=self.expand_aliases_in_llm_output(r)
        r2=r2.replace("0","false")
        r2=r2.replace("1","true")
        #print(r2)
        return str(r2).lower()

    def __init__(self, system_prompt_file, model="gpt-5-mini"):
        #model="gemini-2.5-flash-lite"
        #model="gpt-4.1-mini"
        self.client = stuff.simple_llm(model)
        self.num_parallel=512

        with open(system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        self.client.set_system_prompt(system_prompt)
        self.system_prompt=system_prompt

    def get_batch(self):
        return 2048

    def get_max_size(self):
        return self.client.image_size, self.client.image_size

    def get_stats(self):
        return self.client.get_stats()

    def generate_attributes(self, attrs, jpegs):
        system_prompt=""
        attr_list=[]
        for a in attrs:
            if ":" in a:
                a=a.split(":")[1]
                attr_list.append(a)
        self.attr_list=attr_list
        #    system_prompt+=a+", "
        #system_prompt+="\n</KEY LIST>"

        prompt="""
        Task for vision model:
        1 . Study the supplied image carefully and evaluate **only** the central person selected per system-prompt rules.
        2 . Produce a single JSON object that satisfies the OUTPUT CONTRACT above.
        3 . Return nothing except the minified JSON inside a fenced code-block
        """
        b64_images=[]
        for j in jpegs:
            b64_images.append(base64.b64encode(j).decode('utf-8'))
        s=[]
        for b in b64_images:
            a={"b64_image":b, "system_prompt":system_prompt, "prompt":prompt}
            s.append(a)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            responses = list(executor.map(self.process_images, s))
        del s
        return responses

    """
Gpt-5                FP     Aug 13 Overall        TP=256.0  FP=21.7 FN=41.4  p=0.922 r=0.861 F=0.890 cost/Minf($): 1166.86
Gpt-5-mini           FP     Aug 13 Overall        TP=261.0  FP=27.1 FN=38.2  p=0.906 r=0.872 F=0.889 cost/Minf($): 228.19
Gpt-4.1-mini         FP     Aug 13 Overall        TP=250.0  FP=29.4 FN=50.0  p=0.895 r=0.833 F=0.863 cost/Minf($): 560.50
Gemini2.5-flash-lite FP     Aug 13 Overall        TP=251.0  FP=28.0 FN=48.5  p=0.900 r=0.838 F=0.868 cost/Minf($): 172.94
---------------------
gpt-5 (minimalR)       Aug10  Overall        TP=261.0  FP=35.9 FN=37.5  p=0.879 r=0.874 F=0.877 cost/Minf($): 3737.98
gpt-5-mini (minimalR)  Aug10  Overall        TP=261.0  FP=33.9 FN=37.5  p=0.885 r=0.874 F=0.880 cost/Minf($): 789.60
gpt-5-nano (minimalR)  Aug10  Overall        TP=228.0  FP=67.3 FN=71.5  p=0.772 r=0.761 F=0.767 cost/Minf($): 166.43
gpt-5                   Aug8  Overall        TP=252.0  FP=35.3 FN=47.4  p=0.877 r=0.842 F=0.859 cost/Minf($): 18922.00
gpt-5-mini              Aug8  Overall        TP=261.0  FP=43.3 FN=38.5  p=0.858 r=0.872 F=0.865 cost/Minf($): 3163.15
gpt-5-nano              Aug8  Overall        TP=243.0  FP=36.3 FN=56.5  p=0.870 r=0.811 F=0.840 cost/Minf($): 1142.16
gpt-4.1-mini            Aug8  Overall        TP=264.0  FP=35.1 FN=35.2  p=0.883 r=0.882 F=0.882 cost/Minf($): 803.15
gpt-4.0-mini            Aug8  Overall        TP=254.0  FP=44.0 FN=45.0  p=0.852 r=0.850 F=0.851 cost/Minf($): 722.36
Gemini2.5-flash-lite    Aug8  Overall        TP=257.0  FP=26.9 FN=41.5  p=0.905 r=0.861 F=0.883 cost/Minf($): 253.64


gpt-4.1-mini      Jul18 Overall              TP=268.0  FP=31.3 FN=32.2  p=0.895 r=0.893 F=0.894 cost/Minf($)': '454.64'


gpt-4.1-nano      Jul18 Overall              TP=240.0  FP=53.2 FN=59.7  p=0.819 r=0.801 F=0.810 cost/Minf($)': '235.87'
gpt-4.1-mini      Jul18 Overall              TP=241.0  FP=36.0 FN=57.7  p=0.870 r=0.807 F=0.837 cost/Minf($)': '454.64'
gpt-4.0-mini      Jul18 Overall              TP=279.0  FP=50.0 FN=20.5  p=0.848 r=0.932 F=0.888 cost/Minf($)': '707.87

gpt-4.0-mini      Jul11 Overall              TP=137.0  FP=28.5 FN=12.7  p=0.828 r=0.915 F=0.869
gpt-4.0-mini      Apr22 Overall              TP=137.0  FP=24.8 FN=13.5  p=0.847 r=0.910 F=0.877
gpt-4.1-nano            Overall              TP=136.0  FP=24.8 FN=14.5  p=0.846 r=0.904 F=0.874
gpt-4.1                 Overall              TP=135.0  FP=24.8 FN=15.5  p=0.845 r=0.897 F=0.870
Gpt-4.1-mini            Overall              TP=124.0  FP=12.6 FN=25.5  p=0.908 r=0.830 F=0.867
Gpt-4.0-mini      Apr15 Overall              TP=136.0  FP=20.8 FN=14.5  p=0.868 r=0.904 F=0.885
Gpt-4o-mini        Apr7 Overall              TP=140.0  FP=23.8 FN=10.5  p=0.855 r=0.930 F=0.891 # adjusted GTs
Gpt-4o-mini        Apr7 Overall              TP=138.0  FP=25.5 FN= 6.0  p=0.844 r=0.958 F=0.898
Gpt-4o-mini (high) Mar5 Overall              TP=135.0  FP=25.5 FN= 9.0  p=0.841 r=0.937 F=0.887
Gpt-4o-mini (low) Mar5  Overall              TP=135.0  FP=26.5 FN= 9.0  p=0.836 r=0.937 F=0.884
Gpt-4o-mini Feb25       Overall              TP=134.0  FP=25.5 FN=10.0  p=0.840 r=0.931 F=0.883 # 'very carefully'
Gpt-4o-mini Feb25       Overall              TP=134.0  FP=27.5 FN=10.0  p=0.830 r=0.931 F=0.877 # separate system prompt
Gpt-4o-mini Feb 8       Overall              TP=133.0  FP=29.5 FN=11.0  p=0.818 r=0.924 F=0.868
Gpt-4o Feb 8            Overall              TP=117.0  FP=21.3 FN=26.0  p=0.846 r=0.818 F=0.832
gpt-4o-2024-11-20       Overall              TP=130.0  FP=28.4 FN=13.5  p=0.821 r=0.906 F=0.861
gpt-4o-mini-2024-07-18  Overall              TP=133.0  FP=27.5 FN=11.0  p=0.829 r=0.924 F=0.874
gpt-4o-mini (16 Feb 25) Overall              TP=132.0  FP=30.5 FN=12.0  p=0.812 r=0.917 F=0.861




    """