from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd


# Load HTML
def get_source_html_coin(url):
    driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver')
    driver.maximize_window()

    try:
        driver.get(url=url)
        time.sleep(1)
        while True:
            find_button = driver.find_element_by_xpath( "//button[@class='button__Button-uwgksy-0 button__ActionButtonStyle-uwgksy-1 hDUyKl btdGGk']")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # scroll to the end of page mazafaka
            time.sleep(2)
            find_button = find_button.click() # show more click
            time.sleep(1)
            print('ckicked')
    except Exception as _ex:
        with open(f'./data/CoinDesk', 'w') as file:
            file.write(driver.page_source)
    finally:
        driver.close()
        driver.quit()


# get items and record them
def get_items_urls(file_path):
    with open(file_path) as file:
        src = file.read()
    soup = BeautifulSoup(src, 'lxml')

    df = {'Name': [],
          'Description': [],
          'Link': [],
          'Date': []}
    
    divTag = soup.find_all("div", class_="card__Meta-sc-3i6u6z-1 fGEoXt")
    for j, tag in enumerate(divTag):
        title = tag.find('h3').find('a').text
        link = tag.find('h3').find('a').get('href')
        description = tag.find('p', class_='description__Description-i3x7s5-0 jIwjwd').text
        data = tag.find('time', class_='card__Datetime-sc-3i6u6z-4 HyNVd').text
            
        df['Name'].append(title)
        df['Description'].append(description)
        df['Link'].append('https://www.coindesk.com' + link)
        df['Date'].append(data)
    return df
    

def main():
    #url = 'https://www.coindesk.com/livewire/'
    
    # to get html data
    #get_source_html_coin(url=url)

    # to extract info from html
    df = get_items_urls(file_path='./data/CoinDesk')

    df = pd.DataFrame(df)
    df.to_csv('./data/CoinDesk_parsed.csv')
    print(df)


if __name__ == '__main__':
    main()
