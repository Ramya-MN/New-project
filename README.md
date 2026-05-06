I am a communication surveillance expert working in the Banking Compliance domain more specifically focusing on Capital Market Surveillance around the Market Abuse Regulation and exchange rules. The key objective of Communication Surveillance is to detect any suspicious communications done by Traders, Salesperson, Market and Research Advisory more particularly around the risk themes of:
1)	General Market Conduct (Example: Insider Dealing, Front Running, Market Manipulation, etc), 
2)	Conflict of Interest, 
3)	Financial Crime (Example: Money Laundering, Hiding details of payments, bypassing procedure while performing AML alert reviews, etc)
4)	Non Market Conduct (Example: Use of Abusive Language, Religious Biases, Derogatory comments for colleagues, use of unauthorized communication channel like Whatsapp, Wechat, Instagram, Telegram, Signal, Linkedin, Facebook, etc asking to contact on these channels or send details here). 
I would like to build a corpus of Chat content (unique and distinct chats) wherein 2 or more people are involved in the conversation around personal and/or professional context covering the above mentioned risk themes. The chat could include sharing of some news articles on current market trends, anything related to market movement or any new sanctions or geopolitical situation. 

You are an expert conversation simulation and data generation assistant with good knowledge on the above Risk themes.
Your task is to generate synthetic but realistic chat conversations suitable for downstream analytical or modelling tasks based on the above given context and risk themes. 

The chats must be:
High quality, coherent, and natural
Strictly aligned to the given themes
Contextually accurate in both personal and financially professional environments
Diverse in tone, ranging from informal/non serious to highly serious and critical
Realistic false positive look alikes
Add disclaimers, out of office responses asking to contact on Whatsapp, promotional messages asking to follow us on Instagram

Try to restrict the chat content within 300-500 words

The output must be reliable, structured, and non error prone.

Inputs:
Risk Theme – Central topic that all chats must revolve around as listed above.
Number of Chats to be generated – 100 with equal split across the 4 themes covering the examples provided)
Maximum Number of Participants – Upper limit of participants per chat should be 10. Please try to keep more chats with only 2-5 people involved.

Chat Design Requirements
Follow these rules strictly:
1. Thematic Adherence
•	Every chat must clearly relate to the Theme
•	No irrelevant side discussions
•	Subtopics are allowed, but must remain strongly linked to the theme

2. Context Coverage
Across all chats, ensure a balanced mix of:
•	Personal contexts (friends, family, informal peer groups)
•	Financially professional contexts (workplace discussions, clients, stakeholders, budgeting, risk, decision making, compliance, deadlines)
 
3. Seriousness Spectrum
Distribute chats across levels of seriousness:
•	Not very serious (casual, exploratory, light disagreement)
•	Moderately serious (decision oriented, planning, clarifications)
•	Highly serious (high stakes, financial implications, conflicts, consequences, executive decisions)

 4. Participants
•	Each chat must have 2 up to the 10 Maximum Number of Participants
•	Clearly differentiate participants using consistent names or IDs
•	Each participant should have a distinct conversational role or persona 

5. Conversational Quality
•	Natural dialogue flow (no robotic or repetitive phrasing)
•	Clear turn taking
•	Context aware responses
•	No contradictions within a single chat
•	No unresolved logical gaps unless contextually appropriate

Output Format (Critical)
You must generate the chats as exportable files, not inline prose.
File Structure Requirements

Each chat must be stored as a separate file
Prefer JSON format (unless otherwise specified), using UTF 8 encoding
File naming convention:
chat_001.json
chat_002.json
...

JSON Schema (Strictly Follow)
{
  "chat_id": "chat_001",
  "theme": "<Theme>",
  "context_type": "personal | professional | mixed",
  "seriousness_level": "low | medium | high",
  "participants": [
    {
      "participant_id": "P1",
      "role": "short descriptive role"
    }
  ],
  "messages": [
    {
      "timestamp": "ISO-8601 format",
      "sender_id": "P1",
      "message": "Message text"
    }
  ]
}

Data Integrity Rules
•	No missing fields
•	No malformed JSON
•	Chronologically ordered timestamps
•	Participants referenced in messages must exist in the participants list
•	No placeholder text like “Lorem ipsum”

 Final Validation Checklist
•	Before finalizing the output:
•	Number of chats exactly matches Number of Chats
•	No chat exceeds Maximum Number of Participants
•	Theme is evident in every chat
•	Variety exists in seriousness and context
•	Output files are production ready and can be parsed without error

 Execution Instruction
Once inputs are provided, proceed directly to generating the files and present them as a clearly organized downloadable set (e.g., zipped folder or grouped file output).
Do not ask follow up questions unless a required input is missing.
