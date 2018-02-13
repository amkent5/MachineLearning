# Beautifulsoup web scraper to build up vocabulary / corpus of Oil and Gas industry jargon

"""
Google search: oil and gas industry jargon
URLS:
        http://www.lmoga.com/resources/oil-gas-101/oil-gas-terminology/		(done)
        https://oilandgasuk.co.uk/glossary/		*** a no go
        http://www.oil150.com/about-oil/oil-gas-dictionary/					(done)
        https://en.wikipedia.org/wiki/Glossary_of_oilfield_jargon
        https://en.wikipedia.org/wiki/List_of_abbreviations_in_oil_and_gas_exploration_and_production
PDFS:
        https://www.pwc.com/gx/en/energy-utilities-mining/pdf/eumcommoditiestradingriskmanagementglossary.pdf
"""

from collections import defaultdict
from bs4 import BeautifulSoup
import urllib

d_data = defaultdict()



#####################################################################
# http://www.lmoga.com/resources/oil-gas-101/oil-gas-terminology/
#####################################################################
r = urllib.urlopen('http://www.lmoga.com/resources/oil-gas-101/oil-gas-terminology/').read()
soup = BeautifulSoup(r)
print type(soup)

# sample
#print soup.prettify()[0:1000]

# by inspecting the source I know all content I want from this page
# is contained in a div with class 'section'
data = soup.find_all("div", class_="section")
for line in data[0]:
        if len(line) > 1:
                text = line.text.encode('utf-8')
                l_text = text.split('\xe2\x80\x93')
                if len(l_text) == 2:
                        d_data[l_text[0].replace('\n', '')] = l_text[1]

print len(d_data)



#####################################################################
# http://www.oil150.com/about-oil/oil-gas-dictionary/
#####################################################################

r2 = urllib.urlopen('http://www.oil150.com/about-oil/oil-gas-dictionary/').read()
soup = BeautifulSoup(r2)
#print soup.prettify()[0:5000]

# inspecting the html source shows me I need the content div
data = soup.find_all("div", {"id": "content"})
data = data[0]
print type(data)

# keys are h3 tags and descriptions are p tags
h3_s = data.findAll('h3')
para_s = data.findAll('p')

# filter out the ones we don't want
h3_ixs = [ i for i, elt in enumerate(h3_s) ][1:]
p_ixs = [ i for i, elt in enumerate(para_s) if 'a href' not in str(elt) ][2:]

# add to lookup dict
for i in range(len(h3_ixs)):
	#print h3_s[h3_ixs[i]].text
	#print para_s[p_ixs[i]].text
	#print '\n'

	d_data[h3_s[h3_ixs[i]].text] = para_s[p_ixs[i]].text



########################################################################################################
# http://www.glossary.oilfield.slb.com/en/Disciplines/All-Disciplines.aspx
########################################################################################################

r3 = urllib.urlopen('http://www.glossary.oilfield.slb.com/en/Disciplines/All-Disciplines.aspx').read()
soup = BeautifulSoup(r3)

# on this occasion I want all of the a tags after the 42nd one
data = soup.find_all('a')[43:]

# iterate through a tags storing the text as the key, then open
# the inner webpage and extract the text as the value
d_terms = defaultdict()
for atag in data:
	
	key = atag.text

	new_url = 'http://www.glossary.oilfield.slb.com/' + atag.get("href")
	read_new_url = urllib.urlopen(new_url).read()
	soup = BeautifulSoup(read_new_url)

	inner_data = soup.find_all("div", class_ = "definition-text")
	#print inner_data
	#print inner_data[0].text
	#print '\n'

	d_terms[key] = inner_data[0].text

for k, v in d_terms.items():
	print k
	print v
	print '\n'
print len(d_terms)




quit()



### Resulting dict
for k, v in d_data.items():
	print k
	print v
	print '\n'
print len(d_data)





