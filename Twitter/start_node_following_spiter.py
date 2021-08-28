"""
三个地址需要更改：212，215，217行
chromedriver.exe地址
浏览器用户密码缓存地址
user.txt用户名列表地址
"""
from pprint import pprint
import queue
import re
import time
import traceback
from typing import List, Dict
import crawlertool as tool
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from queue import Queue
import winsound

duration = 1000  # millisecond
freq = 440  # Hz


class SpiderTwitterAccountInfo(tool.abc.SingleSpider):
    """
    Twitter账号信息爬虫
    """
    def __init__(self, driver):
        self.driver = driver
        
        # 爬虫实例的变量
        self.user_name = None
    """
    格式化按行输出关系：a followers_of_a
    """
    def writedata(self,path:str, data):
        f=open(path,'a+')
        for line in data:
            f.write(line[0])
            f.write("\t")
            f.write(line[1]) 
            f.write("\n")
    def writepair(self,path:str, pair):
        print("写入文件"+path+str(pair))
        f=open(path,'a+')
        f.write(pair[0])
        f.write("\t")
        f.write(pair[1]) 
        f.write("\n")
    def running(self,start_node_list,username,despath) -> List[Dict]:
        #先爬取第一跳数据
        print("name="+username)
        followers_name,following_name=self.getFollowers_and_Following(username)
        num=0
        for followers_pair in followers_name :
            self.writepair(despath+'follower_relation_all.txt',followers_pair) 

    @staticmethod
    def get_twitter_user_name(page_url: str) -> str:
        """提取Twitter的账号用户名称
        主要用于从Twitter任意账号页的Url中提取账号用户名称
        :param page_url: Twitter任意账号页的Url
        :return: Twitter账号用户名称
        """
        if pattern := re.search(r"(?<=twitter.com/)[^/]+", page_url):
            return pattern.group()
    
    
    def get_following(self,start_node_list,user_name):
        save=False
        #根据阈值筛选稠密的粉丝
        actual_url = "https://twitter.com/" + user_name+"/following"
        self.console("开始抓取,实际请求的Url:" + actual_url)
        self.driver.get(actual_url)
        time.sleep(10)

        following_name=[]
 
        #这部分下拉加载所有的粉丝
        old_scroll_height = 0 #表明页面在最上端
        js1 = 'return document.body.scrollHeight'#获取页面高度的javascript语句
        js2 = 'window.scrollTo(0, document.body.scrollHeight)'#将页面下拉的Javascript语句
        while (self.driver.execute_script(js1) > old_scroll_height):#将当前页面高度与上一次的页面高度进行对比
            old_scroll_height = self.driver.execute_script(js1)#获取到当前页面高度
            self.driver.execute_script(js2)#操控浏览器进行下拉
            time.sleep(2)#空出加载的时间
            self.driver.execute_script(js2)#操控浏览器进行下拉
            time.sleep(2)#空出加载的时间
            self.driver.execute_script(js2)#操控浏览器进行下拉
            time.sleep(8)#空出加载的时间

            #正式爬取数据
           
            #self.writedata('L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\NLP-Twitter-main\\follower_relation.txt',followers_name)
            try:
                for i in range(1, 1000):
                    following=self.driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[2]/section/div/div/div[{}]/div/div/div/div[2]/div[1]/div[1]/a/div/div[2]/div/span'.format(i))
                    #time.sleep(random.randrange(1,3,1))
                    #去除字符串开始的换行和@
                    str1=following.text
                    str1=str1[1:]
                    str1=str1.strip("\n")
                    following_name.append(str1)
                    print("当前following数量为："+str(len(following_name)))
                   
            except Exception as e:
                print(e)
                # print("--get following of ",user_name)

            
        print("爬取了所有following,共有"+str(len(following_name)))
        return following_name
        
    def get_startnode_following(self,despath,start_node_list):
        #失败列表
        faillist=[]
        
        #循环读取文件中的粉丝名
        for startnode in start_node_list:
             
                startnode=startnode.strip("\n")
                print(startnode)
                #获取关注列表，并根据saveflag决定是否落盘
                try:
                    followinglist=self.get_following(start_node_list,startnode)
                    print("爬取following列表:"+startnode)
                    for following in followinglist :
                        relation_pair=[following,startnode]
                        self.writepair(despath+'startnode_follower_relation.txt',relation_pair)

                except Exception as e:
                    print("爬取失败"+startnode+"加入失败队列")
                    print(e)
                    faillist.append(startnode)
        #失败节点写入失败文件夹
        failf=open(despath+'fail_nodes.txt',"a+")
        for line in faillist:
            failf.write(line+"\n")
        failf.close()

    

#筛选主函数
def spidermain_filter(start_node_list):
    winsound.Beep(freq, duration)
    #初始化webdriver
    driverOptions = Options()
    #导入缓存数据
    driverOptions.add_argument(r"user-data-dir=C:\\Users\\HUAWEI\\AppData\\Local\\Google\\Chrome\\User Data") 
    driverOptions.add_experimental_option("excludeSwitches", ['enable-automation', 'enable-logging'])
    #对应的chromedriver路径
    #driverOptions.add_argument('--headless')
    driver = webdriver.Chrome(executable_path=r"L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\chromedriver_win32\\chromedriver.exe",options=driverOptions)

    #
    SpiderTwitterAccountInfo(driver).get_startnode_following(".\\Limit200\\",start_node_list)

    driver.quit()


#————————————————开始筛选
#获取对应类型的社交网络
if __name__ == "__main__":
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
    # 
    spidermain_filter(start_node_list)


