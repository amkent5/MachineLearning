### Scrape Oil and Gas terminology to create a corpus
# this scrapes 4,913 webpages

import time
import urllib
import pickle
from collections import defaultdict
from bs4 import BeautifulSoup

r = urllib.urlopen('http://www.glossary.oilfield.slb.com/en/Disciplines/All-Disciplines.aspx').read()
soup = BeautifulSoup(r)

# inspected page source and found I want all of the a tags after the 43rd one
data = soup.find_all('a')[43:]

# iterate through a tags storing the text as the key, then open
# the inner webpage and extract the text as the value / description
i = 0
d_terms = defaultdict()
restart_time = True
try:
	for atag in data:
		
		key = atag.text.strip()

		inner_url = 'http://www.glossary.oilfield.slb.com/' + atag.get("href")
		read_inner_url = urllib.urlopen(inner_url).read()
		soup = BeautifulSoup(read_inner_url)

		inner_data = soup.find_all("div", class_ = "definition-text")
		#print inner_data[0].text
		#print '\n'

		d_terms[key] = inner_data[0].text.strip()

		i += 1
		if restart_time:
			t0 = time.time()
			restart_time = False		
		
		if not i % 10:
			print 'Processed %i webpages' % i
			print 'Time taken: ', time.time() - t0, '\n'
			restart_time = True

except KeyboardInterrupt:
	print 'Interrupted'
	for k, v in d_terms.items():
		print k
		print v
		print '\n'
	print len(d_terms)

	# serialise
	with open('d_oil_and_gas_terms.pickle', 'wb') as handle:
		pickle.dump(d_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)


for k, v in d_terms.items():
	print k
	print v
	print '\n'
print len(d_terms)

# serialise
with open('d_oil_and_gas_terms.pickle', 'wb') as handle:
	pickle.dump(d_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)