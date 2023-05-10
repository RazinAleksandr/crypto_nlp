from bs4 import BeautifulSoup
from selenium import webdriver
import time
import datetime
from datetime import timedelta
import pandas as pd


# Load HTML
def get_source_html_coin(url):
    driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver')
    driver.maximize_window()

    try:
        driver.get(url=url)
        time.sleep(5)
        while True:
            for elem in driver.find_elements_by_xpath('.//time[@class = "post-card__date"][@data-testid = "post-card-pulished-date"]'):
                if 'FEB' in elem.text:
                    print(elem.text)
                    with open(f'./data/CoinTelegraph_07_05', 'w') as file:
                        file.write(driver.page_source)
                    return
                else:
                    continue
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 500);") # scroll to the end of page mazafaka
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight - window.innerHeight - 1000);") # scroll to the end of page mazafaka
            #driver.execute_script("window.scrollTo(0, 200);")
            time.sleep(0.1)
    except Exception as _ex:
        print(f'Exception: {_ex}')

    finally:
        driver.close()
        driver.quit()


# get items and record them
def get_items_urls(file_path):
    with open(file_path) as file:
        src = file.read()
    soup = BeautifulSoup(src, 'lxml')

    df = {'Name': [],
          'Link': [],
          'Date': []}
    divTag = soup.find_all("div", class_="post-card")
    for tag in divTag:
        try:
            name = tag.find('a', class_='post-card__title-link').get('title').strip()
            link = tag.find('a', class_='post-card__title-link').get('href')
            data = tag.find('footer', class_='post-card__footer').find('time').text.strip()

            today = datetime.date.today()
            yesterday = today - timedelta(days = 1)
            if 'minutes' in data or 'hours' in data:
                data = yesterday ## today if before 00:00
            else:
                month = data.split(',')[0].split(' ')[0]
                month = '05' if month == 'MAY' else '04'
                day = data.split(',')[0].split(' ')[-1]
                year = data.split(',')[-1].strip()
                data = datetime.datetime.strptime(year+'-'+month+'-'+day, '%Y-%m-%d').date()
            
            df['Name'].append(name)
            df['Link'].append('https://cointelegraph.com' + link)
            df['Date'].append(data)
        except Exception as e:
            continue
    return df
    

def main():
    #url = 'https://cointelegraph.com/'
    
    # to get html data
    #get_source_html_coin(url=url)

    # to extract info from html
    df = get_items_urls(file_path='./data/CoinTelegraph_07_05')

    df = pd.DataFrame(df)
    df.to_csv('./data/CoinTelegraph_parsed_07_05.csv')
    print(df)


if __name__ == '__main__':
    main()
