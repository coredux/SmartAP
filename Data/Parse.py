from lxml import etree

'''
parsing the xml file and getting the useful content
'''
def retrieve_from_xml(xml_content):
    try:
        root = etree.fromstring(xml_content)
        for doc in root.xpath("//document"):
            if doc.text is not None:
                yield doc.text.strip()
    except:
        print "exception at retrieve from xml"
