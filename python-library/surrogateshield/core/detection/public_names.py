"""
detection/public_names.py — names that are public, not personal (audit I12).

Three small gazetteers used by the relation gate (relation_gate.py):

* PUBLIC_PEOPLE — widely known public figures. A NER PERSON in this list is
  kept verbatim unless the message ties it to the user ("my coworker Michael
  Jordan" is still masked).
* PUBLIC_ORGS — global companies, brands, products, platforms, public
  programmes and software. Such an ORG is kept verbatim unless the message
  ties it to a person ("I work at Microsoft" is still masked).
* NOT_NAMES — greetings, closings and capitalised function words that NER
  tags as PERSON/ORG/GPE ("Regards", "Bonjour", "Como").

The lists are general world knowledge, written independently of any
evaluation split; they are deliberately short. Matching is case-insensitive
on the whole entity (possessive "'s" stripped).
"""

from __future__ import annotations


def _set(block: str) -> frozenset:
    return frozenset(
        line.strip().lower()
        for chunk in block.split("\n")
        for line in chunk.split("|")
        if line.strip()
    )


PUBLIC_PEOPLE = _set("""
Earl Grey | Lady Grey
Abraham Lincoln | Adele | Adolf Hitler | Albert Einstein | Alexander Hamilton
Alexander the Great | Alexandria Ocasio-Cortez | Andy Warhol | Angela Merkel
Ariana Grande | Aristotle | Arnold Schwarzenegger | Barack Obama | Beethoven
Ludwig van Beethoven | Benjamin Franklin | Bernie Sanders | Beyoncé | Beyonce
Bill Clinton | Bill Gates | Billie Eilish | Bob Dylan | Bob Marley | Boris Johnson
Brad Pitt | Bruce Lee | Bruce Springsteen | Cardi B | Charles Darwin | Charlie Chaplin
Che Guevara | Cleopatra | Confucius | Cristiano Ronaldo | Dalai Lama | Dante
Dante Alighieri | David Beckham | Donald Trump | Drake | Dwayne Johnson | Ed Sheeran
Elizabeth II | Queen Elizabeth | Elon Musk | Elvis Presley | Emmanuel Macron
Eminem | Franklin D. Roosevelt | Franklin Roosevelt | Frida Kahlo | Galileo
Galileo Galilei | Gandhi | Mahatma Gandhi | George Washington | George W. Bush
Greta Thunberg | Harry Styles | Hillary Clinton | Homer | Isaac Newton
J. K. Rowling | J.K. Rowling | JK Rowling | J. R. R. Tolkien | J.R.R. Tolkien
Jane Austen | Jeff Bezos | Jennifer Lopez | Jesus | Jesus Christ | Joe Biden
John F. Kennedy | John Lennon | Julius Caesar | Justin Bieber | Justin Trudeau
Kamala Harris | Kanye West | Karl Marx | Kim Jong Un | Kim Kardashian | Kobe Bryant
Lady Gaga | LeBron James | Leonardo da Vinci | Leonardo DiCaprio | Lionel Messi
Mahatma | Malala Yousafzai | Mao Zedong | Marie Curie | Mark Zuckerberg
Martin Luther King | Martin Luther King Jr. | Marilyn Monroe | Michael Jackson
Michael Jordan | Michelangelo | Mozart | Wolfgang Amadeus Mozart | Muhammad Ali
Napoleon | Napoleon Bonaparte | Narendra Modi | Neil Armstrong | Nelson Mandela
Nikola Tesla | Oprah | Oprah Winfrey | Pablo Picasso | Picasso | Plato | Pope Francis
Princess Diana | Rihanna | Roger Federer | Ronald Reagan | Rosa Parks | Sam Altman
Satya Nadella | Serena Williams | Shakespeare | William Shakespeare | Sigmund Freud
Snoop Dogg | Socrates | Stephen Curry | Steph Curry | Stephen Hawking | Stephen King
Steve Jobs | Steven Spielberg | Sundar Pichai | Taylor Swift | Tiger Woods
Tim Cook | Tom Brady | Tom Cruise | Tom Hanks | Usain Bolt | Vincent van Gogh
Van Gogh | Vladimir Putin | Volodymyr Zelensky | Walt Disney | Warren Buffett
Winston Churchill | Xi Jinping | Zendaya | Charles III | King Charles | Mark Twain
Ernest Hemingway | George Orwell | Leo Tolstoy | Fyodor Dostoevsky | Charles Dickens
Virginia Woolf | Toni Morrison | Maya Angelou | Edgar Allan Poe | Agatha Christie
Alan Turing | Ada Lovelace | Grace Hopper | Linus Torvalds | Tim Berners-Lee
Thomas Edison | Henry Ford | Jack Ma | Warren Buffet | Noam Chomsky | Carl Sagan
Neil deGrasse Tyson | Freddie Mercury | David Bowie | Paul McCartney | Mick Jagger
Kendrick Lamar | Bad Bunny | Shakira | BTS | Messi | Ronaldo | Pelé | Pele
Diego Maradona | Kylian Mbappé | Kylian Mbappe | Novak Djokovic | Rafael Nadal
Simone Biles | Michael Phelps | Shohei Ohtani | Babe Ruth | Wayne Gretzky
Kevin Durant | Giannis Antetokounmpo | Patrick Mahomes | Lewis Hamilton
Max Verstappen | Virat Kohli | Sachin Tendulkar | Shah Rukh Khan | Amitabh Bachchan
Jackie Chan | Hayao Miyazaki | Emperor Meiji | Genghis Khan | Peter the Great
Catherine the Great | Queen Victoria | Henry VIII | Louis XIV | Joan of Arc
Simón Bolívar | Simon Bolivar | Charlemagne | Augustus | Marcus Aurelius
Joseph Stalin | Vladimir Lenin | Fidel Castro | Theodore Roosevelt | Thomas Jefferson
John Adams | Ulysses S. Grant | Dwight D. Eisenhower | Richard Nixon | Jimmy Carter
George H. W. Bush | Margaret Thatcher | Rishi Sunak | Keir Starmer | Olaf Scholz
Jacinda Ardern | Pedro Sánchez | Giorgia Meloni | Recep Tayyip Erdoğan
Benjamin Netanyahu | Mohammed bin Salman | Lula | Javier Milei | Claudia Sheinbaum
""")

PUBLIC_ORGS = _set("""
Kaiser | Kaiser Permanente | Dyson | Heathrow | Gatwick | Stansted | Schiphol
Alphabet | Google | YouTube | Gmail | Android | Chrome | Microsoft | Windows | Excel
Word | Outlook | Teams | Azure | LinkedIn | GitHub | Xbox | Apple | iPhone | iPad | Mac
MacBook | iCloud | iOS | macOS | Siri | Amazon | AWS | Alexa | Kindle | Prime
Amazon Prime | Whole Foods | Meta | Facebook | Instagram | WhatsApp | Messenger
Threads | Oculus | Netflix | Nvidia | NVIDIA | Tesla | SpaceX | Twitter | X | OpenAI
ChatGPT | Anthropic | Claude | Gemini | Bard | DeepMind | Mistral | Hugging Face
IBM | Intel | AMD | Qualcomm | Samsung | Sony | PlayStation | Nintendo | Switch | LG
Huawei | Xiaomi | Lenovo | Dell | HP | Hewlett-Packard | Asus | Acer | Cisco
Oracle | Salesforce | Adobe | Photoshop | SAP | VMware | Zoom | Slack | Discord
Telegram | Signal | Snapchat | TikTok | ByteDance | Reddit | Pinterest | Tumblr
Spotify | Uber | Lyft | Airbnb | DoorDash | Instacart | Grubhub | PayPal | Venmo
Zelle | Cash App | Stripe | Square | Shopify | eBay | Etsy | Walmart | Target
Costco | Home Depot | Lowe's | Best Buy | Kroger | Walgreens | CVS | Starbucks
McDonald's | Burger King | Subway | Chipotle | Domino's | KFC | Coca-Cola | Coke
Pepsi | PepsiCo | Nike | Adidas | Puma | Zara | H&M | IKEA | Toyota | Honda
Ford | GM | General Motors | Chevrolet | BMW | Mercedes | Mercedes-Benz | Audi
Volkswagen | Porsche | Hyundai | Kia | Nissan | Subaru | Volvo | Ferrari
Boeing | Airbus | Delta | United | American Airlines | Southwest | Lufthansa
Emirates | Ryanair | FedEx | UPS | USPS | DHL | Verizon | AT&T | T-Mobile | Comcast
Xfinity | Spectrum | Vodafone | Netgear | Linksys | TP-Link | Fitbit | Garmin
Disney | Disney+ | Pixar | Marvel | Warner Bros | HBO | Hulu | Paramount
Visa | Mastercard | American Express | Amex | Discover | JPMorgan | JPMorgan Chase
Chase | Bank of America | Wells Fargo | Citi | Citibank | Goldman Sachs
Morgan Stanley | Capital One | Charles Schwab | Schwab | Fidelity | Vanguard
BlackRock | Robinhood | Coinbase | Binance | Bitcoin | Ethereum | Berkshire Hathaway
Medicare | Medicaid | Social Security | IRS | FBI | CIA | NASA | FDA | CDC | WHO
NHS | UN | United Nations | EU | European Union | NATO | DMV | TSA | USCIS
Python | Java | JavaScript | TypeScript | Rust | Go | Golang | Ruby | PHP | Kotlin
Swift | Scala | Perl | Haskell | SQL | PostgreSQL | Postgres | MySQL | SQLite
MongoDB | Redis | Kafka | Elasticsearch | Docker | Kubernetes | Terraform | Ansible
Linux | Ubuntu | Debian | Fedora | Red Hat | CentOS | Nginx | Apache | Node.js | Node
React | Angular | Vue | Svelte | Next.js | Django | Flask | FastAPI | Rails
Ruby on Rails | Spring | Laravel | TensorFlow | PyTorch | pandas | NumPy | Jupyter
Git | GitLab | Bitbucket | Jira | Confluence | Notion | Trello | Asana | Figma
Canva | Dropbox | Box | OneDrive | Google Drive | Wikipedia | Stack Overflow
Cloudflare | Heroku | Vercel | Netlify | DigitalOcean | Twilio | SendGrid | Mailchimp
HubSpot | Zendesk | Okta | Auth0 | Wordpress | WordPress | Wix | Squarespace
Excel | PowerPoint | Word | OneNote | Google Docs | Google Sheets | Gmail | Yahoo
Bing | DuckDuckGo | Firefox | Mozilla | Safari | Edge | Opera | Steam | Epic Games
Roblox | Minecraft | Fortnite | Valve | Riot Games | Twitch | Peloton | Strava
Duolingo | Coursera | Udemy | Khan Academy | Harvard | MIT | Stanford | Oxford
Cambridge | Yale | Princeton | Berkeley | Roth IRA | 401(k) | NBA | NFL | MLB | NHL
FIFA | UEFA | Premier League | Olympics | Super Bowl | World Cup | Grammy | Oscars
SFR | Orange | Telefónica | Movistar | Deutsche Telekom | Banco do Brasil | Itaú
Nubank | Santander | BBVA | HSBC | Barclays | Lloyds | Revolut | Monzo | Wise
N26 | ING | BNP Paribas | Société Générale | Deutsche Bank | Commerzbank | UBS
Credit Suisse | Alibaba | Tencent | WeChat | Alipay | Baidu | JD.com | Taobao
Rakuten | Paytm | Flipkart | Reliance | Tata | Infosys | Wipro | Aramco | Shell
BP | ExxonMobil | Chevron | TotalEnergies | Siemens | Bosch | Philips | Unilever
Nestlé | Nestle | Procter & Gamble | P&G | Johnson & Johnson | Pfizer | Moderna
AstraZeneca | Novartis | Roche | Merck | Bayer | GSK | Kaiser Permanente
UnitedHealthcare | Aetna | Cigna | Blue Cross | Blue Cross Blue Shield | Humana
Planned Parenthood | Red Cross | Salvation Army | Goodwill | Wikipedia | Reuters
BBC | CNN | Fox News | New York Times | Washington Post | Wall Street Journal
Bloomberg | Forbes | The Guardian | Economist | NPR | PBS | ESPN
""")

NOT_NAMES = _set("""
Regards | Best | Best regards | Kind regards | Warm regards | Thanks | Thank you
Thx | Cheers | Sincerely | Yours | Respectfully | Hi | Hello | Hey | Dear | Greetings
Bonjour | Bonsoir | Salut | Merci | Cordialement | Hola | Gracias | Saludos | Buenos
Buenas | Oi | Olá | Ola | Obrigado | Obrigada | Atenciosamente | Hallo | Danke
Grüße | Gruß | Ciao | Grazie | Namaste | Como | Cómo | Qué | Que | Porque | Pourquoi
Comment | Wie | Warum | Was | Wo | Por favor | Please | Router | Modem | Wifi | WiFi
irl | lol | lmao | tbh | imo | imho | idk | btw | omg | ngl | fyi | brb | afaik | smh
""")

# Given names that are also ordinary English words (virtues, gems, flowers,
# months, seasons, trades, verbs). On its own, without anything that points
# to a person, such a word is the word ("rhymes with Joy", "Amber or Jade",
# "Bill of Sale").
WORD_NAMES = _set("""
Madeleine | Joy | Hope | Faith | Charity | Grace | Mercy | Honor | Honour | Victory | Destiny
Patience | Prudence | Constance | Harmony | Melody | Serenity | Trinity | Liberty
Justice | Journey | Chance | Angel | Glory | Bliss | Felicity | Verity | Unity
Amber | Jade | Pearl | Ruby | Crystal | Jewel | Opal | Garnet | Coral | Diamond
Daisy | Rose | Lily | Iris | Violet | Ivy | Holly | Heather | Hazel | Poppy | Fern
Willow | Olive | Jasmine | Laurel | Myrtle | Blossom | Saffron | Sage | Rosemary
April | May | June | August | Summer | Autumn | Winter | Spring | Dawn | Sky | Rain
Storm | River | Brook | Ocean | Star | Sunny | Misty | Snow | Frost | Stormy
Hunter | Baker | Cooper | Carter | Taylor | Porter | Mason | Fisher | Archer
Banner | Page | Penny | Ginger | Candy | Honey | Cherry | Peaches | Story
Bill | Mark | Will | Pat | Sue | Rob | Art | Frank | Grant | Chase | Drew | Gene
Rich | Bob | Jack | Max | Bud | Buck | Duke | Earl | King | Prince | Major | Royal
Cliff | Glen | Dale | Forest | Rock | Stone | Flint | Reed | Heath | Wood | Sterling
Matt | Matte | Rusty | Lance | Miles | Price | Hunt | Ward | Marsh | Rocky
""")
