from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        print("Encountered an end tag :", tag)

    def handle_data(self, data):
        print("Encountered some data  :", data)

parser = MyHTMLParser()
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')

html_text = open("./processed/faculty/cornell/http_^^www.cs.cornell.edu^Info^Department^Annual94^Faculty^Salton.html.txt").read()

parser.feed(html_text)