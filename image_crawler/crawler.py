#!/usr/bin/env python3
"""
Simon Stålenhag Image Crawler
Downloads all high-resolution images from simonstalenhag.se
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from pathlib import Path
import logging

import os

ROOT_DIR = os.path.dirname('/'.join(os.path.abspath(__file__).split('/')[:-1]))


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StalenhagCrawler:
    def __init__(self, base_url="https://www.simonstalenhag.se/", download_dir=ROOT_DIR + "/images"):
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        self.session = requests.Session()
        # Set a user agent to be polite
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create download directory
        self.download_dir.mkdir(exist_ok=True)
        
    def get_page_content(self, url):
        """Fetch and parse a web page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_high_res_images(self, soup):
        """Extract high-resolution image URLs from the page"""
        high_res_urls = set()
        
        # Find all links that point to high-resolution images
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Look for links to 4k images or other high-res patterns
            if ('4k/' in href and href.endswith('.jpg')) or href.endswith('_big.jpg'):
                full_url = urljoin(self.base_url, href)
                high_res_urls.add(full_url)
                logger.info(f"Found high-res image: {href}")
        
        # Also check for any direct image links in src attributes
        for img in soup.find_all('img', src=True):
            src = img['src']
            # Check if this is already a high-res image
            if ('4k/' in src and src.endswith('.jpg')) or src.endswith('_big.jpg'):
                full_url = urljoin(self.base_url, src)
                high_res_urls.add(full_url)
        
        return high_res_urls
    
    def download_image(self, url, filename=None):
        """Download a single image"""
        try:
            if filename is None:
                filename = os.path.basename(urlparse(url).path)
            
            file_path = self.download_dir / filename
            
            # Skip if file already exists
            if file_path.exists():
                logger.info(f"File already exists: {filename}")
                return True
            
            logger.info(f"Downloading: {filename}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded: {filename}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    def crawl_section(self, section_url):
        """Crawl a specific section of the website"""
        logger.info(f"Crawling section: {section_url}")
        
        soup = self.get_page_content(section_url)
        if not soup:
            return []
        
        # Extract high-resolution image URLs
        image_urls = self.extract_high_res_images(soup)
        
        # Download images
        downloaded = []
        for url in image_urls:
            # Add a small delay to be polite to the server
            time.sleep(0.5)
            
            if self.download_image(url):
                downloaded.append(url)
        
        return downloaded
    
    def crawl_by_sections(self):
        """Crawl all sections from the main page"""
        logger.info("Starting to crawl all sections")
        
        # Get main page to find section links
        main_soup = self.get_page_content(self.base_url)
        if not main_soup:
            logger.error("Could not fetch main page")
            return
        
        # Find section links
        section_links = []
        for link in main_soup.find_all('a', href=True):
            href = link['href']
            # Look for section pages (typically .html files)
            if href.endswith('.html') and href != 'index.html':
                section_url = urljoin(self.base_url, href)
                section_links.append(section_url)
        
        # Also crawl the main page itself
        section_links.insert(0, self.base_url)
        
        logger.info(f"Found {len(section_links)} sections to crawl")
        
        all_downloaded = []
        for section_url in section_links:
            downloaded = self.crawl_section(section_url)
            all_downloaded.extend(downloaded)
            
            # Longer delay between sections
            time.sleep(2)
        
        logger.info(f"Crawling complete! Downloaded {len(all_downloaded)} images")
        return all_downloaded
    

def download_images():
    # Create crawler instance
    crawler = StalenhagCrawler()
    downloaded = crawler.crawl_by_sections()
    print(f"\nDownload complete!")
    print(f"Downloaded {len(downloaded) if downloaded else 0} images")
    print(f"Images saved to: {crawler.download_dir}")