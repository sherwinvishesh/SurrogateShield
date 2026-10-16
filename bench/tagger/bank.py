"""Phrase bank for the PIITagger's training messages (V3 §3.3 items 1–4).

Everything here is written for this file: no sentence comes from a dataset
or a user. Three kinds of text:

- ``BODIES``: PII-free requests per task (the carrier), with slots for hard
  negatives that the tagger must leave alone (label O): ``{PUBLIC}`` a widely
  known person, ``{BRAND}`` a company or product named as a topic,
  ``{PLACE}`` a place named as a topic, ``{FICTION}`` a character's name in a
  story or game, ``{IDENT}`` a code identifier, ``{LIB}`` a library, ``{ACRO}``
  an acronym, ``{NUM}`` / ``{YEAR}`` / ``{DATE}`` / ``{MONEY}`` /
  ``{VERSION}`` numbers that are not an age or a birth date, ``{LANG}`` a
  language.
- ``OWN``: sentences that tie a value to the user (``{v}``), per type, in the
  layouts Phase 3 placed values in (intro, inline); ``OTHER_PERSON`` for a
  private third party (a colleague, a child), still PERSON.
- Field names for forms, JSON / YAML / ``.env`` blocks and signatures.
- ``CUE_*`` (data VERSION 3): a name without a cue, as real messages carry
  one (a sentence's start, a possessive, a lower-case label, an aside, a bare
  sign-off, a title with or without a space, a pair, a list); and
  ``FICTION_BODIES`` / ``ESSAY_BODIES``, the same shapes where the name is a
  character's, an author's or a public figure's (label O).

The values themselves never come from here: ``data.py`` draws them from the
training half of the identity pools (``identities.identity(pool="train")``).
"""

from __future__ import annotations

from typing import Dict, List

# ── carriers ─────────────────────────────────────────────────────────────────

GREETINGS = ["Hi", "Hello", "Hey", "Hi there", "Hello!", "Hey there", "Good morning", "Good evening", "Hiya",
             "Dear assistant", "Hey ChatGPT", "Hi Claude", "Yo", "Greetings", "Morning", "Hello again"]
CLOSINGS = ["Thanks!", "Thank you.", "Thanks in advance.", "Cheers", "Much appreciated!", "Thanks a lot",
            "Any help is appreciated.", "Let me know if you need more details.", "Thank you so much!", "Best",
            "Kind regards", "Regards", "Best wishes", "Many thanks", "Thx", "ty", "Appreciate it."]
SIGN_OFFS = ["Best regards,", "Kind regards,", "Regards,", "Thanks,", "Cheers,", "Best,", "Sincerely,",
             "Warm regards,", "Many thanks,", "All the best,", "Yours faithfully,", "Thank you,", "--", "Respectfully,"]

BODIES: Dict[str, List[str]] = {
    "writing": [
        "Can you help me write a cover letter for a junior data analyst position? I have two years of experience with {LIB} and SQL.",
        "Please proofread this paragraph and make it sound more professional: we are sorry for the delay, the shipment got stuck and we will refund the {MONEY} fee.",
        "Write a short wedding toast for my best friend. Keep it under 200 words and make it a little funny.",
        "I need a resignation letter. I've been at the company for {NUM} years and want to leave on good terms.",
        "Rewrite this to be more concise: In order to facilitate the process of onboarding, it is necessary that all new hires complete the training.",
        "Draft a LinkedIn summary for a product manager who moved from marketing into tech.",
        "Can you write a poem about autumn in the style of {PUBLIC}?",
        "Help me write a complaint email about a broken washing machine that was delivered on {DATE}.",
        "Write a thank-you note to my team for finishing the {ACRO} migration ahead of schedule.",
        "I'm writing a fantasy novel. The hero, {FICTION}, has to cross a frozen sea. Can you describe the scene?",
        "Give me five catchy titles for a blog post about remote work.",
        "Write a short story where a detective named {FICTION} solves a case in {PLACE}.",
        "Summarise the main arguments of {PUBLIC}'s most famous essay in plain language.",
        "Improve the flow of my personal statement for a master's programme in public health.",
        "Write a product description for a reusable water bottle that keeps drinks cold for 24 hours.",
        "Can you turn these bullet points into a friendly newsletter intro? - new office plants - team lunch on Friday - {ACRO} training next week",
        "Make this email sound less passive aggressive: As I said in my last three emails, the report is late.",
        "Write a eulogy for my grandfather, who loved fishing and old jazz records.",
        "Draft an out-of-office reply for the holidays, back on {DATE}.",
        "I need a bio for a conference program, about 80 words, third person.",
        "Write a limerick about a cat who works in {BRAND} customer support.",
        "Help me write a speech for my daughter's graduation party.",
        "Can you write a review of {BRAND}'s new headphones? I liked the sound but the battery is weak.",
        "Paraphrase this sentence five different ways: the meeting has been moved to next Tuesday.",
        "Write the opening chapter of a mystery set on a train from {PLACE} to {PLACE}.",
        "Can you write a reference letter for a former intern? She was reliable and great with {LIB}.",
        "Write dialogue between {FICTION} and {FICTION}, two rival chefs on a cooking show.",
        "Write a birthday message for a coworker who is turning {NUM}, keep it light.",
        "Edit my essay intro on climate policy in {PLACE}; it feels too long.",
        "Compose a short apology to a customer whose order {NUM} arrived damaged.",
    ],
    "coding": [
        "Why does this throw a KeyError?\n\n```python\ndef {IDENT}(rows):\n    out = {}\n    for r in rows:\n        out[r['id']] += 1\n    return out\n```",
        "How do I read a CSV with {LIB} and drop rows where the price column is empty?",
        "My {BRAND} build fails with `exit code 137`. What does that mean?",
        "Explain the difference between `let`, `const` and `var` in JavaScript.",
        "Write a SQL query that returns the top {NUM} customers by total order value in {YEAR}.",
        "I get `TypeError: Cannot read properties of undefined (reading 'map')` in my React component. Here is the code:\n\n```jsx\nexport default function {IDENT}({ items }) {\n  return <ul>{items.map(i => <li key={i.id}>{i.name}</li>)}</ul>;\n}\n```",
        "How can I speed up this loop in {LIB}?\n```python\nfor i in range(len(df)):\n    df.loc[i, 'total'] = df.loc[i, 'qty'] * df.loc[i, 'price']\n```",
        "What's the best way to structure a Flask app with blueprints?",
        "Can you convert this bash script to Python? It renames every .jpg file in a folder by date.",
        "Docker compose keeps saying the port is already allocated. How do I find which process uses port {NUM}?",
        "Write a regex that matches dates like 2024-03-15 but not 2024-13-40.",
        "Explain what a closure is with a small example in {LANG}.",
        "Help me write unit tests for this function:\n```python\ndef {IDENT}(a, b):\n    return a / b if b else None\n```",
        "Why is my Kubernetes pod stuck in CrashLoopBackOff? The logs say `connection refused`.",
        "How do I set up a GitHub Actions workflow that runs pytest on every pull request?",
        "My git rebase went wrong and now I have conflicts everywhere. How do I abort it?",
        "What does `{IDENT}` do in this snippet? `const {IDENT} = useMemo(() => compute(data), [data]);`",
        "Write a function in {LANG} that checks if a string is a palindrome.",
        "How do I connect to a PostgreSQL database from Node with connection pooling?",
        "The {LIB} version {VERSION} broke my imports. How do I pin the previous version?",
        "Refactor this to use async/await:\n```js\nfetch(url).then(r => r.json()).then(data => {IDENT}(data)).catch(console.error)\n```",
        "Explain Big-O of quicksort in the worst case and why.",
        "I'm getting a CORS error when calling my API from localhost:3000. How do I fix it on the server?",
        "Write a Dockerfile for a Python {VERSION} app that uses poetry.",
        "What is the difference between {BRAND} and {BRAND} for hosting a small web app?",
        "How do I parse JSON in Go into a struct with nested fields?",
        "My {ACRO} request returns 403 but the token looks valid. What could cause that?",
        "Write a bash one-liner that counts lines in all .py files under src/.",
        "Why does `{IDENT}.append(x)` return None in Python?",
        "Optimise this SQL: SELECT * FROM orders WHERE YEAR(created_at) = {YEAR};",
        "How do I mock {LIB} calls in pytest?",
        "In Rust, why can't I borrow `self.{IDENT}` as mutable more than once?",
    ],
    "qa": [
        "Who was {PUBLIC} and why are they famous?",
        "What is the capital of {PLACE}?",
        "How does photosynthesis work, in simple terms?",
        "When did {PUBLIC} win the Nobel Prize?",
        "What's the difference between a virus and a bacterium?",
        "Why is the sky blue?",
        "How many people live in {PLACE}?",
        "Explain the causes of World War I in a few paragraphs.",
        "What did {PUBLIC} say about democracy?",
        "Is {BRAND} a good company to invest in long term?",
        "What are the main exports of {PLACE}?",
        "How tall is Mount Everest and who first climbed it?",
        "What does {ACRO} stand for?",
        "Compare {BRAND} and {BRAND} in terms of market share.",
        "What happened in {YEAR} that changed the internet?",
        "Can you explain inflation like I'm five?",
        "Who wrote 'Pride and Prejudice' and what is it about?",
        "What is the best time of year to visit {PLACE}?",
        "How do vaccines train the immune system?",
        "What's the population density of {PLACE} compared to {PLACE}?",
        "Why did {PUBLIC} resign?",
        "List the planets in order from the sun.",
        "What's the boiling point of water at high altitude?",
        "Tell me three facts about {PUBLIC}.",
        "How does {BRAND}'s recommendation algorithm work?",
        "What language is spoken in {PLACE}?",
        "Explain quantum entanglement without equations.",
        "How long does it take light from the Sun to reach Earth?",
        "What is {NUM} percent of {NUM}?",
        "Who painted the Mona Lisa, and when?",
    ],
    "advice": [
        "I've had a headache for {NUM} days and ibuprofen isn't helping. Should I see a doctor?",
        "My landlord won't return my deposit of {MONEY}. What are my options?",
        "How should I prepare for a job interview at a big tech company like {BRAND}?",
        "I want to start running but my knees hurt. Any tips?",
        "Is it better to pay off my credit card or put money into savings first?",
        "My dog keeps barking at night. How do I train him to stop?",
        "How do I negotiate a higher salary without sounding greedy?",
        "I'm moving to {PLACE} for work. What should I know about renting there?",
        "What can I do about insomnia without medication?",
        "My teenage son spends all day gaming. How do I talk to him about it?",
        "Should I lease or buy a car if I drive about {NUM} miles a month?",
        "I feel burned out at work. How do I bring it up with my manager?",
        "How do I dispute a charge of {MONEY} on my bank statement?",
        "What are the symptoms of low iron?",
        "My visa expires on {DATE}. How early should I apply for a renewal?",
        "Can I claim my home office on my taxes?",
        "How do I deal with a coworker who takes credit for my work?",
        "I got a parking fine on {DATE} but the sign was hidden. How do I appeal?",
        "What's a reasonable budget for a week in {PLACE}?",
        "My insurance denied my claim. What should I write in the appeal?",
        "Is it normal for a baby to sleep 16 hours a day?",
        "How do I start investing with only {MONEY} a month?",
        "I think I'm being underpaid. How do I find out what others earn?",
        "What should I pack for a hiking trip in {PLACE}?",
        "How can I improve my credit score quickly?",
        "My partner and I argue about chores. Any advice?",
        "Should I see a dermatologist for a mole that changed shape?",
        "How do I write a will without a lawyer?",
    ],
    "translation": [
        "Translate into {LANG}: I would like to book a table for two at eight o'clock.",
        "How do you say 'where is the train station' in {LANG}?",
        "Please translate this email into {LANG} and keep it formal.",
        "Translate to English: Je voudrais changer la date de mon rendez-vous.",
        "What's the {LANG} word for 'thank you very much'?",
        "Translate this into {LANG}: The package will arrive on {DATE}.",
        "Can you translate the following into Spanish? We are happy to confirm your reservation.",
        "Translate to German: Please find the attached documents for my application.",
        "Is this {LANG} sentence correct? 'Ich habe gestern ins Kino gegangen.'",
        "Translate the following letter into {LANG}, keeping names and numbers as they are.",
        "How do I politely decline an invitation in {LANG}?",
        "Translate into French: I am writing to update my contact details.",
        "Please translate this form into {LANG} for my grandmother.",
        "Give me the {LANG} translation and a pronunciation guide.",
    ],
    "roleplay": [
        "Let's play a game. You are {FICTION}, a pirate captain, and I'm a stowaway on your ship.",
        "Pretend you're a medieval innkeeper named {FICTION}. I walk in soaked from the rain.",
        "Roleplay as my job interviewer for a marketing role. Ask me one question at a time.",
        "You are a dungeon master. Start a campaign in the city of {FICTION}.",
        "Act as a travel agent and help me plan a week in {PLACE}.",
        "Let's do a dialogue where you are {PUBLIC} explaining your work to a child.",
        "Be my study buddy and quiz me on the French Revolution.",
        "Pretend to be a grumpy robot called {FICTION} who secretly loves cats.",
        "You are a detective in 1920s {PLACE}. I'm a witness. Begin.",
        "Play the role of a customer support agent for {BRAND}. I'm calling about a refund.",
        "Let's roleplay: I'm a new student and you're the headmaster, {FICTION}.",
        "Continue the story: {FICTION} opened the door and saw the dragon asleep on the gold.",
        "You are an AI from the year 3000. Describe a typical morning.",
    ],
    "business": [
        "Draft an email to a client explaining that the project deadline moves to {DATE}.",
        "Can you write a proposal outline for a {MONEY} marketing campaign?",
        "Create an agenda for a 30-minute weekly team sync.",
        "Write a polite payment reminder for invoice #{NUM}, which was due last week.",
        "Help me write a job ad for a part-time bookkeeper.",
        "Summarise these meeting notes into action items: budget approved, hiring paused, Q3 review moved.",
        "What KPIs should a small e-commerce store track?",
        "Write a cold outreach email to a potential partner in the logistics industry.",
        "Prepare a SWOT analysis for a coffee shop opening near a university.",
        "How should I price a SaaS product aimed at freelancers?",
        "Draft a press release announcing our new office in {PLACE}.",
        "Write a follow-up email after a sales call where the client asked about {BRAND} integration.",
        "Help me reply to a supplier who raised prices by {NUM} percent.",
        "Create a simple onboarding checklist for new employees.",
        "Write a LinkedIn post announcing that we closed our seed round.",
        "Draft terms for a referral program that pays {MONEY} per signup.",
        "Our {ACRO} report is due on {DATE}. Can you suggest a structure?",
        "Write an apology to customers about yesterday's outage.",
        "Turn this into a professional quote: website redesign, 4 pages, {MONEY}, two rounds of revisions.",
    ],
    "other": [
        "Give me a recipe for vegetarian lasagna for six people.",
        "What are some fun things to do on a rainy weekend?",
        "Recommend five books like {PUBLIC}'s novels.",
        "Plan a three-day itinerary for {PLACE}.",
        "What's a good workout routine for beginners at home?",
        "Suggest names for a golden retriever puppy.",
        "How do I get red wine out of a carpet?",
        "Make me a weekly meal plan with a {MONEY} budget.",
        "What should I watch tonight if I liked Breaking Bad?",
        "Tell me a joke about programmers.",
        "Help me plan a surprise party for {NUM} people.",
        "What's the weather usually like in {PLACE} in {MONTH}?",
        "Create a packing list for a beach holiday.",
        "How do I care for a fiddle-leaf fig?",
        "What are good gift ideas for a dad who likes woodworking?",
        "Write a grocery list for tacos.",
        "Explain the rules of cricket briefly.",
        "Help me choose between {BRAND} and {BRAND} for a new phone.",
        "What games can we play at a family reunion?",
        "Can you make a study schedule for my exams starting {DATE}?",
    ],
}

# Bodies that are only negatives: no value is ever placed in these messages.
NEGATIVE_ONLY = [
    "{PUBLIC} and {PUBLIC} met in {YEAR}. What did they talk about?",
    "Compare {BRAND}, {BRAND} and {BRAND} for a startup on a budget.",
    "| Name | Role | Year |\n|---|---|---|\n| {PUBLIC} | physicist | {YEAR} |\n| {PUBLIC} | writer | {YEAR} |\n\nAdd a column for nationality.",
    "class {IDENT}:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n\nHow do I add validation to this?",
    "What's the difference between {ACRO} and {ACRO}?",
    "Write a quiz with 5 questions about {PUBLIC}.",
    "In the novel, {FICTION} betrays {FICTION}. Why do you think the author did that?",
    "user_name = input('Name: ')\nuser_email = input('Email: ')\nprint(f'Hello {user_name}')\n\nHow do I store these in SQLite?",
    "Fill in the template: Dear [Name], thank you for contacting [Company]. Your ticket number is [ID].",
    "My favourite players are {PUBLIC} and {PUBLIC}. Who had the better career?",
    "Which is older, {PLACE} or {PLACE}?",
    "Name,Email,Phone\n<name>,<email>,<phone>\nHow do I import this header-only CSV into {BRAND}?",
    "The variables {IDENT}, {IDENT} and {IDENT} are undefined after the refactor.",
    "Rank these by revenue: {BRAND}, {BRAND}, {BRAND}.",
    "Hi! Can you explain what {LIB} does and when I'd use it over {LIB}?",
    "{ACRO} vs {ACRO}: which should I learn first?",
    "We moved from {BRAND} to {BRAND} last year and our costs dropped {NUM}%.",
    "Write a haiku about {PLACE} in winter.",
    "Who would win in a debate, {PUBLIC} or {PUBLIC}?",
    "Is it fine to use 'John Doe' and 'Jane Roe' as placeholder names in a demo?",
]

# ── hard-negative vocabularies (label O) ─────────────────────────────────────

BRANDS = """AWS|Azure|Google Cloud|Microsoft|Apple|Netflix|Spotify|Tesla|Amazon|Salesforce|Shopify|Stripe|OpenAI|
Meta|IBM|Oracle|SAP|Adobe|Notion|Slack|Zoom|Figma|GitHub|GitLab|Docker|Kubernetes|PostgreSQL|MySQL|MongoDB|Redis|
Excel|PowerPoint|Photoshop|Unity|Unreal Engine|Blender|WordPress|Twitter|Instagram|TikTok|LinkedIn|YouTube|Reddit|
WhatsApp|Telegram|Discord|Uber|Airbnb|Coca-Cola|Nike|Toyota|BMW|Samsung|Sony|Nintendo|PlayStation|Xbox|NASA|IKEA|
Heroku|Vercel|Netlify|Cloudflare|Firebase|Supabase|Jira|Trello|Asana|Dropbox|Google Drive|OneDrive|Canva|Mailchimp|
HubSpot|Zendesk|Twilio|PayPal|Visa|Mastercard|Starbucks|McDonald's|Walmart|Target|Costco|Intel|AMD|Nvidia|Qualcomm|
Android|iOS|Windows 11|macOS|Ubuntu|Linux|Chrome|Firefox|Safari|Edge|Gmail|Outlook|Teams|Kindle|Pixel|iPhone|
Galaxy S24|MacBook Air|ThinkPad|Raspberry Pi|Arduino|Peloton|Fitbit|Garmin|Duolingo|Coursera|Udemy|Khan Academy|
Wikipedia|ChatGPT|Claude|Gemini|Copilot|Midjourney|Stable Diffusion|Hugging Face|Snowflake|Databricks|Tableau|
Power BI|QuickBooks|Xero|Revolut|Monzo|Wise|Robinhood|Vanguard|Fidelity|Lufthansa|Ryanair|Emirates|Volkswagen|
Honda|Ford|Hyundai|Lego|Pepsi|Heineken|Zara|H&M|Adidas|Puma|Rolex|Bose|Logitech|Dell|HP|Lenovo|Asus|Acer""".replace(
    "\n", "").split("|")

LIBRARIES = """pandas|NumPy|scikit-learn|PyTorch|TensorFlow|Keras|React|Vue|Angular|Svelte|Django|Flask|FastAPI|
Express|Next.js|Nuxt|Spring Boot|Rails|Laravel|Tailwind|Bootstrap|jQuery|lodash|axios|requests|BeautifulSoup|
Selenium|Playwright|pytest|Jest|Mocha|Celery|SQLAlchemy|Prisma|Sequelize|Hibernate|Matplotlib|seaborn|Plotly|
spaCy|NLTK|transformers|LangChain|OpenCV|Pillow|Redux|GraphQL|Apollo|Socket.IO|Electron|Flutter|React Native|
Pydantic|Poetry|webpack|Vite|Babel|ESLint|Prettier|Terraform|Ansible|Helm|Kafka|RabbitMQ|Airflow|dbt|Spark|
Hadoop|Elasticsearch|Grafana|Prometheus|Nginx|Apache|Gunicorn|uvicorn|Streamlit|Gradio|XGBoost|LightGBM""".replace(
    "\n", "").split("|")

ACRONYMS = """API|SQL|CRM|ERP|KPI|ROI|SaaS|B2B|GDPR|HIPAA|SOC 2|ISO 27001|HTML|CSS|JSON|YAML|REST|HTTP|TCP|UDP|DNS|
VPN|SSH|SSO|OAuth|JWT|CI/CD|AWS|GCP|ML|AI|NLP|LLM|GPU|CPU|RAM|SSD|USB|HDMI|PDF|CSV|XML|SEO|UX|UI|QA|HR|CEO|
CTO|CFO|MBA|PhD|MSc|BSc|IELTS|TOEFL|SAT|GRE|GMAT|NHS|FDA|WHO|IMF|NATO|UN|EU|OKR|SLA|MVP|PRD|RFC|IDE|SDK|CLI|
ORM|MVC|SPA|PWA|CDN|LAN|WAN|IoT|AR|VR|ETL|BI|OCR|PII|ADHD|IBS|BMI|IVF|PTSD|COVID-19|ETA|FAQ|RSVP|ASAP|FYI""".replace(
    "\n", "").split("|")

LANGUAGES = ["Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Korean", "Mandarin", "Hindi",
             "Arabic", "Dutch", "Swedish", "Polish", "Turkish", "Vietnamese", "Indonesian", "Greek", "Russian",
             "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C#", "Kotlin", "Swift", "Ruby", "PHP"]

# Places named as topics: countries, regions and landmarks. A city a value could
# be (identities.COUNTRIES / EXTRA_CITIES) is never one of these (data.py checks).
PLACES = """France|Japan|Brazil|Canada|Australia|Germany|Italy|Spain|Mexico|India|China|Egypt|Kenya|Nigeria|Peru|
Chile|Argentina|Norway|Sweden|Finland|Iceland|Greece|Turkey|Thailand|Vietnam|Indonesia|the Philippines|
South Korea|New Zealand|Portugal|Morocco|Ghana|Ethiopia|Colombia|Ireland|Scotland|Wales|Poland|Austria|
Switzerland|Belgium|Denmark|Croatia|Hungary|the Netherlands|Paris|Tokyo|Rome|London|Berlin|Madrid|Lisbon|Vienna|
Prague|Budapest|Athens|Istanbul|Cairo|Nairobi|Bangkok|Singapore|Hong Kong|Seoul|Beijing|Shanghai|Mumbai|Delhi|
Sydney|Melbourne|New York|Los Angeles|San Francisco|Boston|Miami|Vancouver|Montreal|Buenos Aires|
Rio de Janeiro|Mexico City|Lima|Bogotá|Cape Town|Marrakech|Dubai|Reykjavik|Oslo|Stockholm|Copenhagen|Helsinki|
Amsterdam|Brussels|Zurich|Geneva|Venice|Florence|Barcelona|Kyoto|Bali|the Alps|the Sahara|the Amazon|
Patagonia|the Andes|the Himalayas|Yosemite|Yellowstone|the Grand Canyon|Lake Como|the Scottish Highlands|
Tuscany|Provence|Bavaria|Andalusia|Silicon Valley|the Mediterranean|Scandinavia|the Balkans|Southeast Asia|
the Caribbean|Hawaii|Alaska|California|Texas|Florida|Quebec|Ontario|Queensland|Siberia|Antarctica""".replace(
    "\n", "").split("|")

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]

IDENTS = ["get_user", "parse_rows", "fetchData", "UserProfile", "handleSubmit", "total_price", "orderItems",
          "build_index", "CustomerService", "isValid", "max_retries", "renderList", "api_client", "DataLoader",
          "compute_stats", "useAuth", "config_path", "OrderRepository", "tmp_df", "processQueue", "the_list",
          "make_request", "SessionManager", "cleanup", "normalize_text", "InvoiceItem", "userRef", "load_config",
          "PaymentGateway", "retry_count", "toggleMenu", "db_session", "HttpClient", "row_count", "is_admin"]

# ── values tied to the user ──────────────────────────────────────────────────

# intro / inline sentences; {v} is the value
OWN: Dict[str, List[str]] = {
    "PERSON": [
        "My name is {v}.", "I'm {v}.", "Hi, I'm {v}.", "This is {v}.", "{v} here.", "Hello, my name's {v} and",
        "I am {v}, and", "Name's {v}.", "It's {v} again.", "My full name is {v}.", "I go by {v}.",
        "Please address the letter to me, {v}.", "Sign it as {v}.", "Put my name, {v}, at the bottom.",
        "The account is under my name, {v}.", "You can call me {v}.", "Write it from me ({v}).",
        "I, {v}, hereby confirm the details.", "From: {v}", "my name is {v} btw", "Use my name {v} on the cover page.",
        "Add my name ({v}) to the author line.", "I'm {v}, the new hire in accounts.",
        "As {v}, I'd like to", "The booking is for {v}.", "This is {v} from the third floor.",
        "My legal name is {v} but", "I signed up as {v}.", "Replace [Your Name] with {v}.",
        "my name: {v}", "{v} (that's me)", "I am writing on behalf of myself, {v}.",
    ],
    "EMAIL": [
        "My email is {v}.", "You can reach me at {v}.", "Send the confirmation to {v}.", "email me at {v}",
        "My address for this is {v}.", "Reply to {v} please.", "Contact: {v}", "Add {v} to the mailing list.",
        "I registered with {v}.", "My login email is {v}.", "cc me at {v}", "Please use {v} for invoices.",
        "my e-mail: {v}", "I changed my email to {v}.", "The account email is {v}.", "drop me a line at {v}",
        "Use {v} as the reply-to address.", "My work email is {v}.", "My personal email is {v}.",
    ],
    "PHONE": [
        "My number is {v}.", "Call me on {v}.", "You can text me at {v}.", "My phone is {v}.",
        "Reach me at {v} after 6pm.", "my cell: {v}", "Phone: {v}", "My mobile number is {v}.",
        "Please call {v} if anything changes.", "WhatsApp me on {v}.", "The best number for me is {v}.",
        "my new number is {v}", "I can be reached on {v}.", "Ring me at {v}.", "Tel. {v}",
        "My contact number is {v}.", "Text {v} when it's ready.",
    ],
    "ADDRESS": [
        "I live at {v}.", "My address is {v}.", "Ship it to {v}.", "Please send the documents to {v}.",
        "My home address: {v}", "We just moved to {v}.", "Deliver to {v}.", "Our flat is at {v}.",
        "The property is {v}.", "My billing address is {v}.", "Mail it to me at {v}.", "I'm at {v}.",
        "Return address: {v}", "Postal address: {v}", "The tenancy is for {v}.", "my address is {v}",
        "Put {v} as my address on the form.", "The parcel should go to {v}.",
    ],
    "LOCATION": [
        "I live in {v}.", "I'm based in {v}.", "I'm from {v}.", "We live just outside {v}.",
        "I recently moved to {v}.", "I work in {v}.", "I'm in {v} right now.", "I grew up in {v} and still live there.",
        "My hometown is {v}.", "Here in {v} it's", "I'm a nurse in {v}.", "We're a family of four in {v}.",
        "I'm currently living in {v}.", "located in {v}", "Our office is in {v}.", "I'm studying in {v}.",
        "I commute into {v} every day.", "As someone living in {v},",
    ],
    "ORG": [
        "I work at {v}.", "I'm an analyst at {v}.", "My employer is {v}.", "I'm a student at {v}.",
        "I teach at {v}.", "I've been with {v} for three years.", "I'm applying from {v}.", "Our company, {v},",
        "I run marketing for {v}.", "My manager at {v} asked me to", "I'm the IT lead at {v}.",
        "I study at {v}.", "I just got a job at {v}.", "I'm writing on behalf of {v}.", "I'm an engineer at {v}.",
        "We at {v} would like to", "I'm a nurse at {v}.", "I intern at {v}.", "Company: {v}",
        "I own a small business called {v}.", "My daughter goes to {v}.", "I graduated from {v}.",
    ],
    "AGE": [
        "I'm {v} years old.", "I am {v}.", "I'm {v} and", "As a {v}-year-old,", "I turned {v} last month.",
        "I'm a {v} year old woman.", "I'm a {v} year old man.", "Age: {v}", "I'm {v} now.", "at {v} years of age",
        "Im {v}", "I'm {v}, recently retired.", "I'm {v} and just started running.", "I'm {v} yo",
        "I'll be {v} next week.", "(I'm {v})", "being {v} years old,",
    ],
    "DATE_OF_BIRTH": [
        "My date of birth is {v}.", "I was born on {v}.", "DOB: {v}", "Born {v}.", "My birthday is {v}.",
        "date of birth {v}", "Birth date: {v}", "I was born {v}.", "my DOB is {v}", "D.O.B. {v}",
        "Put {v} as my date of birth.", "born on {v}",
    ],
    "ID": [
        "My ID number is {v}.", "My account number is {v}.", "Card number: {v}", "My policy number is {v}.",
        "Reference: {v}", "Here is my number: {v}", "My member ID is {v}.", "Please use {v} for my file.",
        "My passport number is {v}.", "My social security number is {v}.", "IBAN: {v}", "My card is {v}.",
        "My licence number is {v}.", "My NI number is {v}.", "My NHS number is {v}.", "my student ID is {v}",
        "The payment came from {v}.", "Customer no. {v}", "my number on file is {v}", "ID: {v}",
    ],
    "NETWORK": [
        "My server's IP is {v}.", "I'm connecting from {v}.", "The home router is at {v}.",
        "My public IP is {v}.", "ssh into {v} fails with permission denied.", "The device MAC is {v}.",
        "My VPS at {v} keeps timing out.", "I whitelisted {v} but it still blocks me.", "host: {v}",
        "ping {v} works but curl doesn't.", "My laptop's address is {v}.", "my IP: {v}",
    ],
    "URL": [
        "Here's my profile: {v}", "My portfolio is at {v}.", "You can see my work at {v}.", "Link: {v}",
        "My LinkedIn is {v}.", "Book a slot with me at {v}.", "My GitHub is {v}.", "My site: {v}",
        "I uploaded the file here: {v}", "Check out my page {v}", "My resume is at {v}.",
    ],
    "HANDLE": [
        "My username is {v}.", "Follow me at {v}.", "I'm {v} on Twitter.", "My handle is {v}.",
        "DM me at {v}.", "My Instagram is {v}.", "add me on Discord: {v}", "I post as {v}.",
        "My gamertag is {v}.", "find me at {v}", "my tag is {v}",
    ],
    "CREDENTIAL": [
        "My password is {v}.", "The API key I'm using is {v}.", "Here's the token: {v}", "password: {v}",
        "I set the secret to {v}.", "The key is {v} but it says invalid.", "Login with {v}.",
        "my pass is {v}", "Use this token: {v}", "The admin password is {v}.", "secret={v}",
        "Authorization: Bearer {v}", "curl -H 'Authorization: Bearer {v}' https://api.example.com/v1/me",
        "export OPENAI_API_KEY={v}", "DATABASE_URL=postgres://app:{v}@db.internal:5432/app", "mysql -u root -p{v}",
        "the login is admin / {v}", "-H \"X-Api-Key: {v}\"", "smtp pass {v}", "SMTP_PASSWORD={v}",
        "api_key={v}", "headers = {'Authorization': 'Bearer {v}'}", "ssh root@host, pw {v}",
    ],
}

# A private third party: still PERSON
RELATIONS = ["colleague", "manager", "boss", "son", "daughter", "wife", "husband", "partner", "friend",
             "landlord", "neighbour", "neighbor", "tenant", "client", "coworker", "sister", "brother", "mother",
             "father", "mum", "mom", "dad", "grandmother", "teacher", "doctor", "lawyer", "roommate", "cousin",
             "fiancé", "ex", "supervisor", "team lead", "student", "patient"]
OTHER_PERSON = [
    "My {rel} {v} said it was fine.", "My {rel}, {v}, is helping me with this.",
    "I also need to email my {rel} {v} about it.", "{v} (my {rel}) wants to read it too.",
    "Can you also help me write a message to {v}, my {rel}?", "My {rel} is called {v}.", "I share a flat with {v}.",
    "Address it to {v}.", "Please cc {v} on this.", "Dear {v},", "Hi {v},", "{v} and I are working on this together.",
    "It's for a meeting with {v} from the sales team.", "I'm meeting {v} tomorrow.",
    "{v} will cover for me while I'm away.", "Tell {v} that I'll be late.", "My {rel} {v} recommended you.",
    "It's a gift for my {rel} {v}.", "My {rel} ({v}) has the same problem.", "Ask {v} if you need anything.",
]

# field names (forms, JSON / YAML keys, .env names)
FORM_KEYS: Dict[str, List[str]] = {
    "PERSON": ["Name", "Full name", "Full Name", "Customer name", "Applicant", "Patient", "Employee", "Contact",
               "Name of applicant", "Account holder", "NAME", "name", "Tenant", "Student", "Your name"],
    "EMAIL": ["Email", "E-mail", "Email address", "EMAIL", "email", "Contact email", "Work email", "Mail"],
    "PHONE": ["Phone", "Phone number", "Mobile", "Tel", "Telephone", "Cell", "PHONE", "Contact number", "Mob."],
    "ADDRESS": ["Address", "Home address", "Street address", "Mailing address", "ADDRESS", "Billing address",
                "Shipping address", "Residential address"],
    "LOCATION": ["City", "Location", "Town", "Based in", "City of residence", "Current city", "LOCATION"],
    "ORG": ["Company", "Employer", "Organisation", "Organization", "School", "Institution", "COMPANY", "Workplace"],
    "AGE": ["Age", "AGE", "age", "Age (years)"],
    "DATE_OF_BIRTH": ["Date of birth", "DOB", "D.O.B.", "Birth date", "Birthday", "DATE OF BIRTH", "Born"],
    "ID": ["ID", "ID number", "Account number", "Policy number", "Member ID", "Card number", "SSN", "IBAN",
           "Passport no.", "Reference", "Customer ID", "Licence number"],
    "NETWORK": ["IP", "IP address", "Server IP", "Host", "MAC address", "Gateway"],
    "URL": ["Website", "Portfolio", "LinkedIn", "Profile", "Link", "URL", "GitHub"],
    "HANDLE": ["Username", "Handle", "Twitter", "Instagram", "Discord", "User", "Gamertag"],
    "CREDENTIAL": ["Password", "API key", "Token", "Secret", "PIN", "Passphrase", "Access key"],
}
JSON_KEYS: Dict[str, List[str]] = {
    "PERSON": ["name", "full_name", "fullName", "customer_name", "author", "user", "contact_name", "owner"],
    "EMAIL": ["email", "email_address", "emailAddress", "contact_email", "reply_to", "mail"],
    "PHONE": ["phone", "phone_number", "phoneNumber", "mobile", "tel"],
    "ADDRESS": ["address", "home_address", "street_address", "shipping_address", "billingAddress"],
    "LOCATION": ["city", "location", "town", "hometown"],
    "ORG": ["company", "employer", "organization", "org", "school"],
    "AGE": ["age"],
    "DATE_OF_BIRTH": ["dob", "date_of_birth", "birthDate", "birthday"],
    "ID": ["id_number", "account_number", "ssn", "iban", "card_number", "policy_no", "member_id", "passport"],
    "NETWORK": ["ip", "host", "server_ip", "ip_address", "mac", "gateway"],
    "URL": ["website", "url", "linkedin", "profile_url", "homepage"],
    "HANDLE": ["username", "handle", "twitter", "instagram", "screen_name"],
    "CREDENTIAL": ["password", "api_key", "token", "secret", "access_key", "auth_token"],
}
ENV_KEYS: Dict[str, List[str]] = {
    "CREDENTIAL": ["API_KEY", "SECRET_KEY", "DB_PASSWORD", "AUTH_TOKEN", "ACCESS_TOKEN", "PASSWORD",
                   "SMTP_PASSWORD", "CLIENT_SECRET", "APP_SECRET"],
    "NETWORK": ["DB_HOST", "SERVER_IP", "HOST", "REDIS_HOST", "BIND_ADDRESS", "GATEWAY_IP"],
    "EMAIL": ["ADMIN_EMAIL", "SMTP_USER", "NOTIFY_EMAIL", "FROM_EMAIL", "SUPPORT_CONTACT"],
    "URL": ["WEBHOOK_URL", "PROFILE_URL", "HOMEPAGE"],
    "HANDLE": ["BOT_USERNAME", "TWITTER_HANDLE"],
    "PERSON": ["AUTHOR", "MAINTAINER", "OWNER_NAME"],
}

CODE_WRAPPERS = [
    "```python\n{block}\n```", "```json\n{block}\n```", "```\n{block}\n```", "```yaml\n{block}\n```",
    "{block}", "```js\n{block}\n```", "```bash\n{block}\n```", "```sql\n{block}\n```",
]

# ── ages of named people (V4 §3.1 D) ─────────────────────────────────────────
# The layouts the V4 AGE probe found the tagger reading as another type, or
# not at all: a name, a comma and an age closing a sign-off line, an age after
# a third person's name, in brackets, a Reddit "(29F)". Slots: {n} a full name,
# {f} a first name, {c} a city, {v} an age, {r} a relation, and a second
# person's {f2} / {v2}. Worded apart from ``age_probe.TEMPLATES``: the probe
# measures the layout, not these strings (tests/test_V4_tagger_age_data.py).
AGE_LINES = [
    "Written by {n}, {v}", "Candidate: {n}, {v}", "Volunteer - {n}, {v}", "Guest: {n}, {v}", "Member: {n}, {v}",
    "Prepared by {n}, {v}", "Driver: {n}, {v}", "Author: {n}, {v}", "Participant 3: {n}, {v}",
    "Next of kin: {n}, {v}", "Emergency contact - {n}, {v}", "Player: {n}, {v}", "Interviewee: {n}, {v}",
    "Owner — {n}, {v}", "Requested by {n}, {v}", "Passenger 2: {n}, {v}",
]
AGE_MID = [
    "My {r} {n}, {v}, has had a cough for a week.", "{n}, {v}, will be taking over the account.",
    "Our tenant {n}, {v}, hasn't paid since March.", "I'm helping my {r}, {n}, {v}, with a cover letter.",
    "Yesterday {n}, {v}, slipped on the ice outside the shop.", "The applicant, {n}, {v}, has five years of experience.",
    "Our youngest volunteer, {n}, {v}, organised the whole event.", "Last week {n}, {v}, was promoted to manager.",
    "Is it normal that {n}, {v}, sleeps eleven hours a night?", "We're hiring {n}, {v}, as a part-time tutor.",
    "The witness, {n}, {v}, said the car was red.", "My {r}, {f}, {v}, wants to try climbing.",
]
AGE_BRACKET = [
    "My {r} {f} ({v}) is starting school soon.", "{n} ({v}) has applied for the role.",
    "{n} (age {v}) needs a referral letter.", "My {r} {f} (aged {v}) won't eat vegetables.",
    "Our kids, {f} ({v}) and {f2} ({v2}), share a room.", "Attendee: {n} ({v})", "{n} [{v}] joined the team today.",
    "I look after {f} ({v}) on weekends.", "Can you write a birthday card for {f} ({v})?",
    "Patient {n} ({v}) missed two appointments.", "The new manager, {n} ({v}), starts in May.",
]
AGE_FROM = [
    "{n}, {v}, of {c}", "{n} ({v}), {c}", "{n}, {v} — {c}", "Winner: {n}, {v}, {c}",
    "{n}, {v}, lives in {c} and works nights.", "Profile: {n}, {v}, {c}", "{n}, {v}, {c}, says the bus is always late.",
]
AGE_REDDIT = [
    "I ({v}F) don't know what to do about my roommate.", "{v}m, uni student here.",
    "So my BF ({v}M) said something weird last night.", "[{v}F] Update on my job situation",
    "Me, {v}M, and my GF, {v2}F, are moving in together.", "AITA? I'm {v}F and my sister is {v2}.",
    "My husband ({v}M) and I ({v2}F) can't agree on a budget.", "F{v} here, first post.",
    "My ({v}F) manager keeps changing my shifts.", "{v}/M, need some career advice.",
]
AGE_WORDED = [
    "My {r} {f} turned {v} on Sunday.", "{f}, who is {v}, has asked for help with this.",
    "{n} is {v} and retiring soon.", "My {r} is {v} and has never used email.",
    "{n}, now {v}, still runs every morning.", "We're planning a party because {f} turns {v} next month.",
]
AGE_FORM = [("Member", "Age"), ("Guest", "Yrs"), ("Attendee", "Age (yrs)"), ("Child", "age"),
            ("Kid's name", "Age"), ("Participant", "AGE"), ("Traveller", "Years")]
# forms and lines that only hold a child's age, and the kin a child's age goes with
AGE_CHILDREN = ("Our kids", "Child", "Kid's name")
CHILD_KIN = ["son", "daughter", "sister", "brother", "cousin", "nephew", "niece", "grandson", "granddaughter",
             "stepson", "stepdaughter", "little brother", "little sister"]
# the same shapes, where the number is not an age (label O)
AGE_NEAR = [
    "{n}, {k} tickets", "{n}, Room {k}", "{n}, ext. {k}", "{n}, Desk {k}", "{n}, Table {k}", "{n}, Flat {k}",
    "{n}, Team {k}", "{n}, {YEAR} cohort", "{n}, {k}:30", "{n} ({k} votes)", "{n} (page {k})",
    "Gate {NUM}, platform {k}", "Week {NUM}, day {k}", "Total items, {k}", "{PLACE} {NUM}, {PLACE} {k}",
    "Chapter {NUM}, page {k}", "{LIB} {VERSION}, {k} open issues", "{BRAND} Pro, {k} GB", "Bus {NUM}, stop {k}",
    "Table {NUM}, {k} guests", "Order {NUM}, {k} items", "{FICTION}, chapter {k}",
]

# ── names without a cue (data VERSION 3) ─────────────────────────────────────
# Real messages carry a name with no "my name is": at a sentence's start, as
# a possessive, after a lower-case label, in an aside, alone under a blank
# line as a sign-off, after a title with or without a space, two on a line, a
# list. Slots: {v} / {v2} / {v3} full names, {first} / {last} / {first2}
# their parts, {X} a title and a surname ("Mr.Okafor", "Dr. Okafor"; the span
# holds the title), {job} a job title (O), {tail} a value line under a bare
# sign-off (a phone, an email, or a job). Every name is a PERSON.
CUE_START = [
    "{v} will be joining the call on Monday, can you draft the invite?",
    "{v} asked me to send you the figures before the meeting.",
    "{v} is my manager and wants this by Friday.",
    "{v} from HR sent me this form and I don't understand question 4.",
    "{v} here, following up on my question from yesterday.",
    "{v} needs the letter translated by Thursday.",
    "{v} recommended you, so here goes.",
    "{v} left this voicemail and I can't make out the second half.",
    "{v} and I are splitting the rent; how do we word the agreement?",
    "{v} is the landlord; he wants the deposit paid to a new account.",
    "{v} got the job! Help me write a congratulations note.",
    "{v} has asked for a reference and I need to keep it short.",
    "{v} will pick up the parcel; what should the note say?",
    "{v} is the patient; the appointment is at 9.",
    "{v} wrote this and asked me to check it.",
    "{v} again. Did you get my last message?",
    "{v} (that's my wife) wants a second opinion on this.",
    "{v} is coming to dinner on Saturday and is allergic to nuts.",
    "{v} says the invoice was paid on the 3rd.",
    "{v} is the name on the lease, not mine.",
]
CUE_POSSESSIVE = [
    "{v}'s contract ends next month and I need to write the renewal.",
    "I'm reviewing {v}'s application; is this cover letter any good?",
    "This is {v}'s laptop, I'm borrowing it for the day.",
    "Can you check {v}'s essay for grammar?",
    "{v}'s flight lands at nine; draft a message for the driver.",
    "We're planning {v}'s leaving do.",
    "{v}'s invoice is still unpaid.",
    "I found {v}'s phone on the bus. How do I return it?",
    "Fix the spelling in {v}'s bio below.",
    "{v}'s results came back and the doctor wants a follow-up.",
    "It's {v}'s birthday on Sunday; ideas for a card?",
    "Please summarise {v}'s notes from the meeting.",
    "{v}'s account was charged twice.",
    "I'm {v}'s carer and need to explain the medication schedule.",
    "{v}'s visa interview is next week.",
    "Our neighbour {v}'s fence fell onto our drive.",
]
CUE_LABEL = [
    "the tenant in question: {v}", "for the attention of: {v}", "Attn: {v}", "applicant - {v}", "re: {v}",
    "regarding: {v}", "candidate: {v}", "requested by {v}", "prepared for {v}", "name on the booking: {v}",
    "person to contact: {v}", "sender: {v}", "cc: {v}", "to: {v}", "Attention: {v}", "FAO {v}", "c/o {v}",
    "on behalf of {v}", "signed, {v}", "yours, {v}", "account holder: {v}", "the name on the card is {v}",
    "driver: {v}", "patient name {v}", "submitted by {v}", "written by {v}", "owner {v}", "guest: {v}",
    "for {v}", "attendee {v}", "with {v}", "ref. {v}", "contact person {v}", "the other party is {v}",
    "for the file of {v}", "booking under {v}", "collected by {v}", "insured: {v}", "student {v}",
]
CUE_ASIDE = [
    "(this is {v} btw)", "(it's me, {v})", "({v} speaking)", "(from {v})", "({v}, the one who emailed yesterday)",
    "[{v}]", "(signed {v})", "it's {v} again", "{v} again,", "- {v}", "({v})", "— {v}", "this is {v},",
    "me = {v}", "{v}, here", "({v}, flat 4)", "{v} speaking.", "(sent by {v})",
]
# a name alone under a blank line, with no sign-off phrase
CUE_BARE = ["{v}", "{v}", "{v}", "{v}", "- {v}", "~ {v}", "–{v}", "{v}\nSent from my phone", "{v}\nSent from my iPhone",
            "{v}\n{job}", "{v} | {job}", "{v}\n{tail}", "{v}\n{tail}", "{v}, {job}", "{v}.", "{v} x", "{v} :)"]
JOBS = ["Project Manager", "Office Manager", "Senior Analyst", "Teacher", "Account Executive", "Nurse",
        "Freelance Designer", "PhD student", "Owner", "Sales", "Customer Success", "Operations", "Head of Finance",
        "Parent governor", "Landlord", "Volunteer coordinator", "Student", "Driver", "Site manager", "Receptionist"]
TITLES = ["Mr.", "Mr.", "Mr", "Mrs.", "Mrs", "Ms.", "Ms", "Dr.", "Dr", "Prof.", "Mx.", "Miss", "Herr", "Mme", "Sr."]
CUE_TITLE = [
    "Please address it to {X}.", "{X} will sign on my behalf.", "I'm {X}, the landlord.", "{X} is my GP.",
    "Dear {X},", "my teacher {X} said", "ask for {X} at reception", "this is {X} from the clinic",
    "{X} and I spoke on Tuesday.", "The letter is for {X}.", "{X} here.", "Regards, {X}", "{X} (me) and",
    "Can you reply to {X} for me?", "{X} asked for the report.", "We met {X} at the open day.",
    "cc {X}", "{X} is the one who complained.", "Thanks, {X}", "Say it's from {X}.", "I am {X}.",
    "{X} - that's me - and", "the account is in the name of {X}", "{X} (my husband) and I",
]
# two people on one line: a couple, co-signers, a child and a parent
CUE_PAIR = ["{v} & {v2}", "{v} and {v2}", "{v} & {X}", "{first} & {X}", "{first} and {X}", "{v}, {v2} and family",
            "{v} + {v2}", "{v}/{v2}", "{v} and {first2}", "{first} & {first2} {last}", "{first} & {X}",
            "{v} and {v2} (parents)"]
CUE_LIST = [
    "Attendees:\n- {v}\n- {v2}\n- {v3}", "Guests: {v}, {v2} and {v3}", "Team:\n{v}\n{v2}", "Invited: {v}; {v2}",
    "Participants\n1. {v}\n2. {v2}\n3. {v3}", "Cc: {v}, {v2}", "Players: {v}, {v2}, {v3}",
    "People in the group chat: {v}, {v2}, {v3}", "witnesses: {v} and {v2}", "The tenants are {v} and {v2}.",
    "Speakers:\n* {v}\n* {v2}", "Names for the badges: {v}, {v2}, {v3}", "Travellers: {v}, {v2}",
    "Emergency contacts: {v} / {v2}",
]
CUE_MDFIELD = ["**Full name**: {v}", "**Name**: {v}", "*Name* - {v}", "__Name__: {v}", "**Applicant** {v}",
               "Name — {v}", "NAME {v}", "Name {v}", "**Name:**{v}", "`name`: {v}", "Name:{v}", "name = {v}",
               "**Patient**: {v}", "**Contact** – {v}", "**Full Name**: {v}", "_Name_: {v}"]

# ── fiction, essays, citations: names that are no one's (label O) ────────────
# Scripts and stories with a cue-free character's name ({FICTION2} a first
# and last name, {SPEAKER} an ALL-CAPS speaker label), authors in citations
# ({SURNAME}, {INITIAL}), institutions named after places, public figures as
# topics, third-person ages in a plot. The tagger learns the register, not
# the shape of the name.
FICTION_BODIES = [
    "Write the next scene. {FICTION2} walked into the kitchen and found the letter on the table.",
    "{FICTION2}: Where were you last night?\n{FICTION2}: Out. Why do you care?\n{FICTION2}: Because the car was gone.\n\nContinue this dialogue for another ten lines.",
    "INT. KITCHEN - NIGHT\n\n{SPEAKER}\n(quietly)\nI never meant for any of this.\n\n{SPEAKER}\nThen why did you sign it?\n\nWrite the next page of this screenplay.",
    "{SPEAKER}: (laughing) You can't be serious.\n{SPEAKER}: Dead serious. Pack your things.\n\nMake this exchange funnier.",
    "{FICTION2}'s hands were shaking as she opened the envelope. Continue the story in the same voice.",
    "In my novel, {FICTION2} is a retired detective and {FICTION2} is her estranged brother. Suggest a plot twist.",
    "Chapter 3\n\n{FICTION2} had never liked the sea. Describe the storm from his point of view.",
    "Narrator: The town had one road in and one road out.\n{FICTION}: And we're taking it.\n\nWrite the next exchange.",
    "Rewrite this with more tension: {FICTION2} stood at the door. {FICTION} did not look up.",
    "Characters: {FICTION2} (the mayor), {FICTION2} (her rival), {FICTION} (a journalist). Write the opening scene.",
    "Give me a character sheet for {FICTION2}, a rogue in a D&D campaign.",
    "Write a fan fiction where {FICTION2} and {FICTION2} get stuck in an elevator.",
    "{FICTION2} will be joining the crew in episode four. Draft his first lines.",
    "The protagonist, {FICTION2}, wakes up with no memory. Outline three acts.",
    "{FICTION2} (V.O.): I should have left when I had the chance.\n\nContinue the voiceover.",
    "{SPEAKER}\n(to {FICTION})\nYou knew. You knew the whole time.\n\n{SPEAKER}\nI suspected.\n\nAdd stage directions.",
    "Scene 2. {FICTION2} enters, soaked. {FICTION2} is at the piano.\n{FICTION}: You're late.\n{FICTION}: The bridge was out.\n\nKeep going in this style.",
    "My D&D party: {FICTION2} the paladin, {FICTION2} the warlock and {FICTION} the bard. Write their tavern intro.",
    "Write a monologue for {FICTION2}, a lighthouse keeper who has not spoken to anyone in a year.",
    "{FICTION2} looked at {FICTION2} and said nothing. Make this paragraph sadder.",
    "Give {FICTION2} a backstory. She's the antagonist of a YA fantasy.",
    "Story prompt: {FICTION2}, 34, a courier, finds a parcel addressed to herself.",
    "Can you write the obituary of a fictional character, {FICTION2}, who died at 85 in the last chapter?",
    "Lyrics draft: {FICTION} waits by the river, {FICTION} never comes. Suggest a second verse.",
    "{SPEAKER}: Objection!\n{SPEAKER}: Overruled. Continue, counsellor.\n\nWrite the cross-examination.",
    "Mr. {SURNAME} is the villain of my screenplay. Give him a monologue.",
    "Mrs. {SURNAME} runs the boarding house in my story. Describe her kitchen.",
    "Dr. {SURNAME} has just discovered the formula. Write the scene where she tells {FICTION2}.",
    "Captain {SURNAME} turned the ship around. Continue in the style of a sea novel.",
    "In the poem, {FICTION2} is both the lover and the sea. Explain the metaphor.",
    "Write a short story where {FICTION2} meets her younger self.",
    "The comic's hero, {FICTION2}, loses his powers in issue 12. Pitch issue 13.",
    "Hamlet: To be or not to be.\n{FICTION2}: Definitely be.\n\nContinue this parody.",
    "{SPEAKER} (CONT'D)\nAnd another thing. Nobody leaves this room.\n\n{SPEAKER}\n(standing)\nWatch me.\n\nFinish the scene.",
    "Act II, scene 1. {FICTION2} and {FICTION2} are alone on stage. Write forty lines of blank verse.",
    "{FICTION2} - the detective\n{FICTION2} - the widow\n{FICTION2} - the butler\n\nWrite a whodunit outline with these three.",
    "Fan theory: {FICTION2} was dead the whole time. Argue for and against.",
    "Episode 4 recap: {FICTION2} betrays the crew, {FICTION2} escapes. Write the cold open of episode 5.",
    "{FICTION2}, {FICTION2} and {FICTION2} are the three sisters in my play. Give each a distinct voice.",
    "Text message from {FICTION2} to {FICTION2}, in the story: 'don't come home tonight'. Write her reply.",
]
ESSAY_BODIES = [
    "According to {SURNAME} and {SURNAME} ({YEAR}), remote work raises output by {NUM}%. Is that study credible?",
    "{SURNAME} et al. ({YEAR}) argue that sleep debt is cumulative. Summarise their position.",
    "Cite this properly: {SURNAME}, {INITIAL}. ({YEAR}). The economics of attention. Journal of Behavioural Science, {NUM}({NUM}).",
    "I'm writing about the {PUBLIC} era for an essay. How did {PUBLIC}'s trade policy differ from the previous one?",
    "{PUBLIC}'s approach to leadership is often contrasted with {PUBLIC}'s. Write two paragraphs on that.",
    "The University of {PLACE} and the {PLACE} Institute of Technology both offer this course. Which is better for robotics?",
    "A {NUM}-year-old boy in the story finds a map. Is that too young for the plot to work?",
    "The novel follows a {NUM}-year-old widow in {PLACE}. Write the blurb.",
    "In {YEAR}, {PUBLIC} wrote that the state should wither away. Who was she responding to?",
    "References\n\n{SURNAME}, {INITIAL}. & {SURNAME}, {INITIAL}. ({YEAR}). Title of the book. Publisher.\n{SURNAME}, {INITIAL}. ({YEAR}). Title of the article. Journal, {NUM}({NUM}).\n\nFormat these in APA.",
    "Bank of {PLACE} reported {MONEY} in losses in {YEAR}. Explain what happened.",
    "({SURNAME}, {YEAR}, p. {NUM}) - is this the right way to cite a page in Harvard style?",
    "The {SURNAME} report ({YEAR}) recommended three reforms. List them.",
    "Compare {SURNAME}'s theory of motivation with {SURNAME}'s. Keep it to 300 words.",
    "Who was {PUBLIC}'s main rival, and did they ever meet?",
    "Discuss the {SURNAME} v. {SURNAME} ruling of {YEAR} and its impact on privacy law.",
    "The study followed {NUM} children aged 8 to 12 in {PLACE}. What's wrong with its sampling?",
    "A 5-year-old and a 7-year-old share a room in the book. Is that realistic for the period?",
    "{PUBLIC} was {NUM} when she published her first paper. Write a timeline of her career.",
    "Credits: directed by {FICTION2}, written by {FICTION2}, music by {FICTION2}. Format this as an end-credits roll.",
    "Edit my literature review: {SURNAME} ({YEAR}) found X, while {SURNAME} and {SURNAME} ({YEAR}) found Y.",
    "The firm {SURNAME} & {SURNAME} LLP advised on the deal. What does an LLP mean?",
    "Judge {SURNAME} dismissed the case in {YEAR}. Explain the reasoning in plain English.",
    "Professor {SURNAME}'s lecture on {ACRO} was confusing. Can you explain the basics?",
    "The {SURNAME} family in the TV series owns the vineyard. Who inherits it in season 3?",
    "A 72-year-old man in the case study has a persistent cough. What would a GP consider first?",
    "My essay is on {PUBLIC}. She was born in {YEAR}; what were the key events of her twenties?",
    "{SURNAME} ({YEAR}, p. {NUM}) calls this 'the paradox of choice'. Is that a fair reading?",
    "The {PLACE} School of Economics published a paper by {SURNAME} and {SURNAME}. Summarise it.",
    "Cast: {FICTION2} as the king, {FICTION2} as the fool. Write the playbill blurb.",
]
