from datasets import load_dataset
dataset = load_dataset('gpt-omni/VoiceAssistant-400K')

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HiTZ/Latxa-Llama-3.1-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HiTZ/Latxa-Llama-3.1-70B-Instruct").to('cuda')

def build_prompt(qs, ans):
    prompt = f"""You are a professional translator from English to Basque.
Translate the following conversation between a user and a chatbot accurately and naturally.
After each translation, rate the quality on a scale from 0 to 1 (1 = perfect, 0 = poor).
If the conversation is language-dependent (for example, relies on English idioms, jokes, wordplay, or culture-specific expressions), is about translating text into another language, continues a previous conversation and lacks full context or refers to something not present in the conversation, give it a low score.
Output strictly in this format:
Galdera: <Basque question>
Erantzuna: <Basque answer>
Score: <number>

Examples:

English=
question: How are you today?
answer: I'm doing great, thank you for asking!
Basque=
Galdera: Nola zaude gaur?
Erantzuna: Oso ondo nago, eskerrik asko galdetzeagatik.
Score: 1.0

English=
question: What's the weather like in Bilbao?
answer: It's cloudy and cool today.
Basque=
Galdera: Nolakoa da eguraldia Bilbon?
Erantzuna: Gaur lainotuta eta fresko dago.
Score: 1.0

English=
question: Can you help me with my homework?
answer: Sure, what subject is it?
Basque=
Galdera: Lagundu ahal didazu nire etxeko lanekin?
Erantzuna: Bai, zein ikasgai da?
Score: 1.0

English=
question: I have hungry.
answer: I think you mean "I am hungry."
Basque=
Galdera: Gosea dut.
Erantzuna: Uste dut esan nahi duzula “gosea dut”.
Score: 0.4

English=
question: Break a leg before your performance!
answer: Thanks, I'll do my best!
Basque=
Galdera: Hautsi hanka zure emanaldia baino lehen!
Erantzuna: Eskerrik asko, saiatuko naiz!
Score: 0.2   # Idiom; literal meaning lost in Basque.

English=
question: Why did the scarecrow win an award?
answer: Because he was outstanding in his field!
Basque=
Galdera: Zergatik irabazi zuen soro-zaintzaileak sari bat?
Erantzuna: Bere eremuan nabarmentzen zelako!
Score: 0.2   # Pun-based joke; not meaningful in Basque.

English=
question: Can you translate "Good morning" into Spanish?
answer: It means "Buenos días."
Basque=
Galdera: Itzuli al dezakezu “Egun on” gaztelaniara?
Erantzuna: “Buenos días” esan nahi du.
Score: 0.3   # Translation-related; not a natural Basque dialogue.

English=
question: How do you say "I love you" in Japanese?
answer: You can say "Aishiteru."
Basque=
Galdera: Nola esaten da “Maite zaitut” japonieraz?
Erantzuna: “Aishiteru” esan dezakezu.
Score: 0.3   # Translation content; irrelevant to Basque conversation.

English=
question: It's raining cats and dogs!
answer: Yeah, better grab an umbrella.
Basque=
Galdera: Katua eta txakurra ari da euria egiten!
Erantzuna: Bai, hobe da aterkia hartzea.
Score: 0.2   # Literal translation; idiom doesn't make sense in Basque.

English=
question: That's exactly what you said yesterday.
answer: Yeah, but now I'm sure about it.
Basque=
Galdera: Hori bera esan zenuen atzo.
Erantzuna: Bai, baina orain ziur nago.
Score: 0.3   # Continuation of previous context; unclear standalone meaning.

English=
question: As I mentioned earlier, I already tried that.
answer: Oh, right. Then maybe restart your computer.
Basque=
Galdera: Lehen esan dudan bezala, hori jada saiatu naiz.
Erantzuna: Bai, egia da. Agian berrabiarazi ordenagailua.
Score: 0.3   # Depends on earlier conversation; lacks context.

English=
question: That's not what you told me before!
answer: I changed my mind.
Basque=
Galdera: Hori ez zenuen lehen esan!
Erantzuna: Iritziz aldatu naiz.
Score: 0.3   # Contextual continuation; not a full, self-contained dialogue.

English=
question: Are they more expensive than regular?
answer: Yes, they are more expensive, as they are better.
Basque=
Galdera: Garestiagoak al dira ohikoen aldean?
Erantzuna: Bai, askoz ere garestiagoak, hobeak baitira.
Score: 0.3 # Contextual continuation; refers to something that is not on the conversation. 

Now translate the following:

English=
question: {qs}
answer: {ans}

Basque=
"""
    return [{"role": "user", "content": prompt}]

def translate_example(example):

    messages = build_prompt(example['question'], example['answer'])

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

translated = dataset['train'].select(range(2)).map(
    translate_example,
    # batched=True,
    # batch_size=1,  # you can increase this if GPU has enough memory
    desc="Translating English → Basque"
)

# 6️⃣ Save translated dataset
# translated.save_to_disk("english_basque_translated_dataset")
print("✅ Translated dataset saved to 'english_basque_translated_dataset'")
