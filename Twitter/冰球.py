import twitter_spiter
from base64 import encode
from xml.sax.xmlreader import IncrementalParser
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint
import time
import json
import re
def get_KG(pre,filename):
 
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    sparql.setQuery('''
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    SELECT DISTINCT ?item ?itemLabel ?twittername ?countryLabel WHERE {
        ?item wdt:P118 wd:Q1215892 ;
            wdt:P2002 ?twittername.
        ?item '''+pre+''' ?country .
      
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 3000
    ''')
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert() 

    f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NHL\\"+filename,"a+",encoding='utf-8')
    for result in qres['results']['bindings']:
        
        wikidata_items=result['item']['value'].strip('\n')
        country_of_citizenshipLabel=result['countryLabel']['value'].strip('\n')
        f.write(wikidata_items+"\t"+country_of_citizenshipLabel+"\n")
        pprint(wikidata_items+"\t"+country_of_citizenshipLabel+"\n")

"""_main_:类型，谓词，存储的文件名"""
if __name__ == "__main__":
    filenamelist={}
    #filenamelist["wdt:P27"]="country_of_citizenship.txt"#国籍
    #filenamelist["wdt:P19"]="place_of_birth.txt"#出生地
    #filenamelist["wdt:P69"]="educated_at.txt"#学校
    #filenamelist["wdt:P108"]="employer.txt"#工作单位
    #filenamelist["wdt:P413"]="position_played_on_team.txt"#球队内的位置
    filenamelist["wdt:P54"]="term.txt"#所属球队
    for k,v in filenamelist.items():
        get_KG(k,v)
        time.sleep(10)