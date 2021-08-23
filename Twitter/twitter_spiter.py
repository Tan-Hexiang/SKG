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
import random
import winsound
maxfollowers_num=500
maxfollowing_num=400
min_interaction=5#最低5个
max_interaction=15#15个直接终止
maxfollowing=500#get_following专用，最多浏览这么多就终止
logfile=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\log.txt","w")
logfilter=open("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\logfilter.txt","w")
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
        

    @staticmethod
    def get_twitter_user_name(page_url: str) -> str:
        """提取Twitter的账号用户名称
        主要用于从Twitter任意账号页的Url中提取账号用户名称
        :param page_url: Twitter任意账号页的Url
        :return: Twitter账号用户名称
        """
        if pattern := re.search(r"(?<=twitter.com/)[^/]+", page_url):
            return pattern.group()
    
    def check_Sparse(self,name):
        actual_url = "https://twitter.com/"+name
        self.console("开始抓取,实际请求的Url:" + actual_url)
        self.driver.get(actual_url)
        time.sleep(3)
        self.console(f"开始判断{name}的稀疏度")
        following=0
        followers=0

        #读取关注和被关注数量
        try:
            if label := self.driver.find_element_by_xpath(
                "//*[@id=\"react-root\"]/div/div/div/main/div/div/div/div[1]/div/div/div/div/div[1]/div/div[5]/div[1]/a"):
                following = tool.extract.number(label.text)
            elif label := self.driver.find_element_by_xpath(
                "//*[@id=\"react-root\"]/div/div/div/main/div/div/div/div[1]/div/div/div/div/div[1]/div/div[4]/div[1]/a"):
                following= tool.extract.number(label.text)
            else:
                self.log("Twitter账号:" + name + "|账号正在关注数抓取异常")
                following = 0

            if label := self.driver.find_element_by_xpath(
                "//*[@id=\"react-root\"]/div/div/div/main/div/div/div/div[1]/div/div/div/div/div[1]/div/div[5]/div[2]/a"):
                followers = tool.extract.number(label.text)
            elif label := self.driver.find_element_by_xpath(
                "//*[@id=\"react-root\"]/div/div/div/main/div/div/div/div[1]/div/div/div/div/div[1]/div/div[4]/div[2]/a"):
                followers = tool.extract.number(label.text)
            else:
                self.log("Twitter账号:" + name + "|账号粉丝数抓取异常")
                followers = 0

            self.console(f"followers:{followers}  following:{following}")
        except:
            self.console(f"稀疏度爬取失败，用户名{name}")
            return True
        
        
        #去除小稀疏度的点
        if followers < 100 :
            return False
        elif following < 10 :
            return False
        else :
            return True

    def get_following(self,start_node_list,user_name):
        save=False
        #根据阈值筛选稠密的粉丝
        actual_url = "https://twitter.com/" + user_name+"/following"
        self.console("开始抓取,实际请求的Url:" + actual_url)
        self.driver.get(actual_url)
        time.sleep(10)

        following_name=[]
        num=0
        exitflag=False
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
            time.sleep(5)#空出加载的时间

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
                    
                    #发现连接到初始节点的边，加入返回list，num++，判断是否到阈值
                    if str1 in start_node_list:
                        following_name.append(str1)
                        num=num+1
                        print("当前following 初始node 数量："+str(len(following_name)))
                        #交集大于阈值，退出，save设为true
                        if num >= min_interaction:
                            save=True
                            # 交集大于最大关系数，直接退出
                            if num >max_interaction:
                                exitflag=True
                                break
                    #当浏览了超过一定数量的following，直接退出
                    if len(following_name)>maxfollowers_num :
                        save=False
                        exitflag=True
                        break
                        
            except Exception as e:
                print(e,file=logfilter)
                # print("--get following of ",user_name)

            if exitflag==True:
                print(str(user_name)+"判断结果为："+str(save))
                break

        return following_name,save,num
        
    def filter_followers(self,despath,start_node_list):
        #过滤器，只使用被5整除的粉丝。一类5万左右粉丝使用1万
        nob=0
        #读取列表
        f_all=open(despath+"follower_relation_all.txt","r",encoding='utf-8')
        print("读取粉丝："+despath+"follower_relation_all.txt")
        #循环读取文件中的粉丝名
        for relation_str in f_all.readlines():
            #保留五分之一
            nob=nob+1
            if (nob%5)==0:
                relation=relation_str.split("\t",1)
                follower_name=relation[1].strip("\n")
                print(follower_name)
                #获取关注列表，并根据saveflag决定是否落盘
                interlist,saveflag,internum=self.get_following(start_node_list,follower_name)
                if saveflag == False:
                    print("舍弃:"+follower_name+" 稠密度为："+str(internum),file=logfilter)
                    print("舍弃:"+follower_name+" 稠密度为："+str(internum))
                elif saveflag == True:
                    pprint(interlist)
                    print("保留:"+follower_name+" 稠密度为："+str(internum),file=logfilter)
                    print("保留:"+follower_name+" 稠密度为："+str(internum))
                    for firstnode in interlist :
                        relation_pair=[firstnode,follower_name]
                        self.writepair(despath+'follower_relation.txt',relation_pair)
                    pair=[relation[0].strip('\n'),follower_name]
                    self.writepair(despath+'follower_relation.txt',pair)
                    # 存储粉丝关注的初始节点数
                    pair=[follower_name,str(internum)]
                    self.writepair(despath+'followerName_firstNodeNum.txt',pair)
            

    def getFollowers_and_Following(self, user_name: str) -> List[Dict]:
        """执行Twitter账号信息爬虫
        :param user_name: Twitter的账号用户名称（可以通过get_twitter_user_name获取）
        :return: Json格式的Twitter账号数据
        """
        self.user_name = user_name

        #错误的链接，直接跳过
        try:
            actual_url = "https://twitter.com/" + user_name+"/followers"
            self.console("开始抓取,实际请求的Url:" + actual_url)
            self.driver.get(actual_url)
        except:
            print("cannot open url :"+"https://twitter.com/" + user_name+"/followers")
            return

        time.sleep(10)
        exitflag=False
        followers_name=[]
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
            time.sleep(5)#空出加载的时间
            

            #正式爬取数据
            
            #self.writedata('L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\NLP-Twitter-main\\follower_relation.txt',followers_name)
            try:
                for i in range(1, 1000):
                    followers=self.driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[2]/section/div/div/div[{}]/div/div/div/div[2]/div[1]/div[1]/a/div/div[2]/div/span'.format(i))
                    #time.sleep(random.randrange(1,3,1))
                    #去除字符串开始的换行和@
                    str1=followers.text
                    str1=str1[1:]
                    str1=str1.strip("\n")
                    #被关注者在前面
                    if [user_name,str1] not in followers_name:

                        followers_name.append([user_name,str1])
                        print(str1)
                        print("当前follower数量"+str(len(followers_name)))
                        if len(followers_name)>maxfollowers_num :
                            print(len(followers_name))
                            exitflag=True
                            break
            except Exception as e:
                print(e)
                print("--get followers of ",user_name)

            if exitflag==True:
                print("followers满")
                break
        #self.writedata('L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\NLP-Twitter-main\\follower_relation.txt',followers_name)
        
        # actual_url = "https://twitter.com/" + user_name+"/following"
        # self.console("开始抓取,实际请求的Url:" + actual_url)
        # self.driver.get(actual_url)
        # time.sleep(10)

        # exitflag=False
        following_name=[]
        # #这部分下拉加载所有的粉丝
        # old_scroll_height = 0 #表明页面在最上端
        # js1 = 'return document.body.scrollHeight'#获取页面高度的javascript语句
        # js2 = 'window.scrollTo(0, document.body.scrollHeight)'#将页面下拉的Javascript语句
        # while (self.driver.execute_script(js1) > old_scroll_height):#将当前页面高度与上一次的页面高度进行对比
        #     old_scroll_height = self.driver.execute_script(js1)#获取到当前页面高度
        #     self.driver.execute_script(js2)#操控浏览器进行下拉
        #     time.sleep(10)#空出加载的时间

        #     #正式爬取数据
           
        #     #self.writedata('L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\NLP-Twitter-main\\follower_relation.txt',followers_name)
        #     try:
        #         for i in range(1, 1000):
        #             following=self.driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[2]/section/div/div/div[{}]/div/div/div/div[2]/div[1]/div[1]/a/div/div[2]/div/span'.format(i))
        #             #time.sleep(random.randrange(1,3,1))
        #             #去除字符串开始的换行和@
        #             str1=following.text
        #             str1=str1[1:]
        #             str1=str1.strip("\n")
        #             #被关注者在前面
        #             if [str1,user_name] not in following_name:
        #                 following_name.append([str1,user_name])
        #                 print(str1)
        #                 print("当前following数量："+str(len(following_name)))
        #                 if len(following_name)>maxfollowing_num :
        #                     print(len(following_name))
        #                     exitflag=True
        #                     break
        #     except Exception as e:
        #         print(e)
        #         print("--get following of ",user_name)

        #     if exitflag==True:
        #         print("following满")
        #         break
        # #self.writedata('L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\NLP-Twitter-main\\follower_relation.txt',following_name)
        if len(followers_name)<20 :
            winsound.Beep(freq, duration)
            print("列表加载不全"+user_name,file=logfile)

        return followers_name,following_name

    def running(self,start_node_list,username,despath) -> List[Dict]:
        #先爬取第一跳数据
        print("name="+username)
        followers_name,following_name=self.getFollowers_and_Following(username)
        num=0
        for followers_pair in followers_name :
            self.writepair(despath+'follower_relation_all.txt',followers_pair)
        #     #写入第一跳:判断粉丝的following_list与初始节点列表的交集
        #     following_list=self.get_following(followers_pair[1])
        #     inter=list(set(start_node_list) & set(following_list))
        #     print("比较："+username+"和"+followers_pair[1],file=logfile)
        #     pprint(inter)
        #     print(len(inter))
        #     if len(inter)>min_interaction :
        #         self.writepair(despath+'follower_relation.txt',followers_pair)
        #         print("写入："+followers_pair[1]+" 交集大小为 "+str(len(inter)),file=logfile)
        #         num=num+1
        #     else:
        #         print("舍弃："+followers_pair[1]+" 交集大小为 "+str(len(inter)),file=logfile)
        # print(username+" 满足条件的粉丝数量是："+str(num))
        # for following_pair in following_name :
        #     #写入第一跳
        #     self.writepair(despath+'follower_relation_all.txt',following_pair)
                


def spidermain(start_node_list):
    winsound.Beep(freq, duration)
    #初始化webdriver
    driverOptions = Options()
    #导入缓存数据
    driverOptions.add_argument(r"user-data-dir=C:\\Users\\HUAWEI\\AppData\\Local\\Google\\Chrome\\User Data") 
    driverOptions.add_experimental_option("excludeSwitches", ['enable-automation', 'enable-logging'])
    #对应的chromedriver路径
    #driverOptions.add_argument('--headless')
    driver = webdriver.Chrome(executable_path=r"L:\\社交知识图谱\\联合基金重点项目\\网络爬虫\\NLP-Twitter-main\\chromedriver_win32\\chromedriver.exe",options=driverOptions)
  
    
    # 爬取粉丝信息的部分---------------------------------------------------------------------------------
    # f=open(source_file,'r')
    # qname=queue.Queue(0)
    # for name in f.readlines():
    #     name=name.strip('\n')
    #     qname.put(name)
        
    # while qname.empty() !=True :
    #     name=qname.get()
    #     try:
    #         SpiderTwitterAccountInfo(driver).running(start_node_list,SpiderTwitterAccountInfo.get_twitter_user_name(f"https://twitter.com/{name}").strip('\n'),desti_path)
    #         print("已完成："+str(name))
    #     except Exception as e:
    #         #爬取过程中出现异常，重新尝试
    #         traceback.print_exc()
    #         #qname.put(name)
    #         #print(str(name)+"发生异常："+str(e),file=logfile)

    # 筛选粉丝稠密度的主函数
    #SpiderTwitterAccountInfo(driver).filter_followers("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NBA\\",start_node_list)
    SpiderTwitterAccountInfo(driver).filter_followers("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\NFL\\",start_node_list)
    #SpiderTwitterAccountInfo(driver).filter_followers("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\英超\\",start_node_list)
    #SpiderTwitterAccountInfo(driver).filter_followers("L:\\社交知识图谱\\联合基金重点项目\\数据\\Limit200\\美国棒球联赛\\",start_node_list)


    driver.quit()
        


