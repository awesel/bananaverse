This is my project for the Nano Banana Hackathon from Google DeepMind. I made a little graphic novel generator.

Features
- You can enter a story, or have AI generate a story about a concept. I like to have the AI generate a story where a student and teacher have a conversation about something I want to learn.
- It will generate sprites for scenes and characters. You can override any of these if you wish to include yourself or one of your drawings in the story. This step is what makes the comics have continuity instead of recreating each character in each panel
- It will then generate panels and stitch them together live.

To run:
1. Set up a venv. Run the command below to make your workspace identical to mine.
2. Set your GEMINI_API_KEY and FAL_API_KEY in a .env, or using EXPORT
3. Run 'python server.py' and visit your localhost!

python3 -m venv .venv && source .venv/bin/activate && pip install \
annotated-types==0.7.0 anyio==4.10.0 blinker==1.9.0 cachetools==5.5.2 \
certifi==2025.8.3 charset-normalizer==3.4.3 click==8.2.1 fal_client==0.7.0 \
Flask==3.1.2 google-auth==2.40.3 google-genai==1.33.0 h11==0.16.0 \
httpcore==1.0.9 httpx==0.28.1 httpx-sse==0.4.1 idna==3.10 itsdangerous==2.2.0 \
Jinja2==3.1.6 MarkupSafe==3.0.2 pillow==11.3.0 pyasn1==0.6.1 \
pyasn1_modules==0.4.2 pydantic==2.11.7 pydantic_core==2.33.2 \
python-dotenv==1.1.1 requests==2.32.5 rsa==4.9.1 sniffio==1.3.1 \
tenacity==9.1.2 typing_extensions==4.15.0 typing-inspection==0.4.1 \
urllib3==2.5.0 websockets==15.0.1 Werkzeug==3.1.3
