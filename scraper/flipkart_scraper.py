# scraper/flipkart_scraper.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time, csv, re, random, os
from bs4 import BeautifulSoup

def parse_price(text):
    if not text:
        return None
    return int(''.join(ch for ch in text if ch.isdigit()))

def get_driver(options):
    try:
        return webdriver.Chrome(options=options)
    except Exception:
        try:
            # Try using webdriver-manager to auto-install chromedriver
            from webdriver_manager.chrome import ChromeDriverManager
            return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        except Exception as e:
            raise RuntimeError("Could not start Chrome webdriver. Install chromedriver or webdriver-manager. Original error: " + str(e))

def scrape_search_page(query, page=1):
    url = f"https://www.flipkart.com/search?q={query}&page={page}"

    options = Options()
    options.add_argument("--headless=new")
    # Optional: set a user agent and window size to reduce blocking
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
    options.add_argument("window-size=1920,1080")

    driver = None
    try:
        driver = get_driver(options)
        driver.get(url)

        # Scroll down to load products
        for _ in range(5):  # adjust number of scrolls
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    items = []
    cards = soup.select("div[data-id]")  # product cards
    print(f"Found {len(cards)} product containers")

    for c in cards:
        # try several selectors that Flipkart uses in different layouts
        name = c.select_one("div._4rR01T") or c.select_one("a.s1Q9rs") or c.select_one("a")
        price = c.select_one("div._30jeq3") or c.select_one("div.Nx9bqj") or c.select_one("div._1vC4OE") or c.select_one("div._25b18c")
        rating_el = c.select_one("div._3LWZlK") or c.select_one("div.XQDdHH")
        reviews_el = c.select_one("span._2_R_DZ") or c.select_one("span.Wphh3N")

        if not name:
            continue

        # product name via image alt fallback to avoid UI labels like "Add to Compare"
        img = c.find('img', alt=True)
        if img and img.get('alt'):
            product_name = img['alt'].strip()
        else:
            product_name = name.get_text(strip=True).replace('Add to Compare', '').replace('Bestseller', '').strip()

        # get price text: prefer element, else search for rupee symbol in card
        price_text = None
        if price:
            price_text = price.get_text()
        else:
            m = re.search(r"₹\s*[\d,]+", str(c))
            if m:
                price_text = m.group(0)

        if not price_text:
            print(f"Skipping product (no price found in text): {product_name[:60]}")
            continue

        # parse rating and reviews (fallback to searching card text)
        rating_val = None
        reviews_count = None
        if rating_el:
            m = re.search(r"(\d+(\.\d+)?)", rating_el.get_text())
            if m:
                rating_val = float(m.group(1))
        else:
            # prefer a decimal that is followed by 'Ratings' to avoid capturing screen sizes
            m = re.search(r"(\d+\.\d+)\s*,\s*\d+\s*Ratings", c.get_text())
            if m:
                rating_val = float(m.group(1))
            else:
                m = re.search(r"(\d+\.\d+)\s*Ratings", c.get_text())
                if m:
                    rating_val = float(m.group(1))

        if reviews_el:
            m = re.search(r"(\d[\d,]*)", reviews_el.get_text())
            if m:
                try:
                    reviews_count = int(m.group(1).replace(",", ""))
                except Exception:
                    reviews_count = None
        else:
            m = re.search(r"(\d[\d,]*)\s*Reviews", c.get_text())
            if m:
                try:
                    reviews_count = int(m.group(1).replace(",", ""))
                except Exception:
                    reviews_count = None

        item = {
            "product_name": product_name,
            "price": parse_price(price_text),
            "rating": rating_val,
            "reviews_count": reviews_count,
            "source": "Flipkart",
            "query": query,
            "page": page
        }

        items.append(item)

    return items

def scrape_query(query, max_pages=10, out_csv="data/raw/flipkart_laptops.csv"):
    all_items = []
    for p in range(1, max_pages + 1):
        try:
            items = scrape_search_page(query, page=p)
            print(f"Page {p}: scraped {len(items)} items")
            all_items.extend(items)
            time.sleep(random.uniform(1.5, 3.5))
        except Exception as e:
            print(f"Page {p} error: {e}")
    
    if not all_items:
        print("No items scraped. Check your selectors or query.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_items[0].keys())
        writer.writeheader()
        writer.writerows(all_items)
    print(f"Saved {len(all_items)} rows to {out_csv}")

if __name__ == "__main__":
    scrape_query("laptop", max_pages=10)  # start small to test