"""
发现有twitter链接的谓词：
http://xmlns.com/foaf/0.1/homepage
http://dbpedia.org/ontology/wikiPageWikiLink	
http://dbpedia.org/property/website

wd:Q9448 英超
wd:Q155223 NBA
Q1163715 美国棒球联赛
"""
import twitter_spiter
from base64 import encode
from xml.sax.xmlreader import IncrementalParser
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint
import json
import re
import io
import sys
import time
import random
logf=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\mainlog.txt","w")
def get_twitter_user_name(page_url: str) -> str:
        """提取Twitter的账号用户名称
        主要用于从Twitter任意账号页的Url中提取账号用户名称
        :param page_url: Twitter任意账号页的Url
        :return: Twitter账号用户名称
        """
        if pattern := re.search(r"(?<=twitter.com/)[^/]+", page_url):
            return pattern.group()

"""_main_"""
def get_wikidata(type,username_path,relation_path):
    """
    爬取DBpedia中所有type类的twitter账号姓名，存放于username_path中
    爬取DBpedia中所有type类实体与其推特主页的对应关系，存放于relation_path
    type值参见dbo中的类型量
    """
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    sparql.setQuery('''
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    SELECT DISTINCT ?item ?itemLabel ?twittername ?twitterid WHERE {
        ?item wdt:P106 '''+type+''' ;
            wdt:P2002 ?twittername.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 200
    ''')
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert() 
    
    twitternames=[]
    wikidata_items=[]
    relations_wiki_and_twitter=[]#wikidata item与twittername对应存储
    for result in qres['results']['bindings']:
        #print(result['item']['value'].strip('\n')+"-->"+result['twittername']['value'].strip('\n'))
        wikidata_items.append(result['item']['value'].strip('\n'))
        twitternames.append(result['twittername']['value'].strip('\n'))
        relations_wiki_and_twitter.append([result['item']['value'].strip('\n'),result['twittername']['value'].strip('\n')])

    #将twittername单独落盘
    f1=open(username_path,"a+", encoding='utf-8')
    for user_name in twitternames:
        f1.write(user_name+"\n")
    #将wikidata实体与twittername对应落盘
    f2=open(relation_path,"a+",encoding='utf-8')
    for relation_wiki_and_twitter in relations_wiki_and_twitter:
        f2.write(relation_wiki_and_twitter[0]+"\t"+relation_wiki_and_twitter[1]+"\n")
  

def get_KG(type,pre,filename,typefilepath):
    """
    爬取DBpedia中所有type类的twitter账号姓名，存放于username_path中
    爬取DBpedia中所有type类实体与其推特主页的对应关系，存放于relation_path
    type值参见dbo中的类型量
    """
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    sparql.setQuery('''
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    SELECT DISTINCT ?item ?itemLabel ?twittername ?countryLabel WHERE {
        ?item wdt:P106 '''+type+''' ;
            wdt:P2002 ?twittername.
        ?item '''+pre+''' ?country .
      
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 200
    ''')
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert() 

    f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\"+typefilepath+"\\"+filename,"a+",encoding='utf-8')
    for result in qres['results']['bindings']:
        
        wikidata_items=result['item']['value'].strip('\n')
        country_of_citizenshipLabel=result['countryLabel']['value'].strip('\n')
        f.write(wikidata_items+"\t"+country_of_citizenshipLabel+"\n")
        pprint(wikidata_items+"\t"+country_of_citizenshipLabel+"\n")

"""_main_:类型，谓词，存储的文件名"""
filenamelist={}
filenamelist["wdt:P27"]="country_of_citizenship.txt"#国籍
filenamelist["wdt:P19"]="place_of_birth.txt"#出生地
filenamelist["wdt:P69"]="educated_at.txt"#学校
filenamelist["wdt:P108"]="employer.txt"#工作单位
filenamelist["wdt:P413"]="position_played_on_team.txt"#球队内的位置
typelist={}
# typelist["wd:Q5482740"]="wdQ5482740"#程序员
# typelist["wd:Q33999"]="wdQ33999"#演员
# typelist["wd:Q177220"]="wdQ177220"#歌手
# typelist["wd:Q5716684"]="wdQ5716684"#舞者
# typelist["wd:Q82955"]="wdQ82955"#政治人物
# typelist["wd:Q2309784"]="wdQ2309784"#自行车运动员
typelist["wd:Q9448"]="英超"#American football player橄榄球运动员
typelist["wd:Q155223"]="NBA"
typelist["wd:Q1215884"]="NFL"
typelist["wd:Q1163715"]="美国棒球联赛"
'''属性信息：
篮球棒球都有：position played on team / speciality ：P413   =》结果是列表/字符串
'''
# for ktype,vtype in typelist.items():
#     get_wikidata(ktype,"L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\"+vtype+"\\twittername.txt","L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\"+vtype+"\\twitter_and_wiki.txt")
#     for k,v in filenamelist.items():
#         get_KG(ktype,k,v,vtype)
#         time.sleep(random.randrange(5,20))

#user_followers.spidermain("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\wdQ33999\\twittername.txt","L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\wdQ33999\\")



#获取对应类型的社交网络
start_node_list=[]
f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NBA\\twittername.txt","r",encoding='utf-8')
for name in f.readlines():
    name=name.strip("\n")
    start_node_list.append(name)
f.close()
f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NFL\\twittername.txt","r",encoding='utf-8')
for name in f.readlines():
    name=name.strip("\n")
    start_node_list.append(name)
f.close()
f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NHL\\twittername.txt","r",encoding='utf-8')
for name in f.readlines():
    name=name.strip("\n")
    start_node_list.append(name)
f.close()
f=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\美国棒球联赛\\twittername.txt","r",encoding='utf-8')
for name in f.readlines():
    name=name.strip("\n")
    start_node_list.append(name)
f.close()
#爬取粉丝
twitter_spiter.spidermain(start_node_list,"L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NHL\\twittername.txt","L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NHL\\")

#启动入口
#twitter_spiter.spidermain(start_node_list)


