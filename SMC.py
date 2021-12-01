import tweepy
from tweepy import OAuthHandler
from collections import Counter
from datetime import date
from tweepy import Cursor
import datetime
import pandas as pd 
import csv
import json
import os
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import community
from sklearn.cluster import KMeans
from operator import itemgetter
st.set_option('deprecation.showPyplotGlobalUse', False)
#cd Desktop\Trimester 2 2021\SMC\Assignment
#streamlit run SMC.py

min_max_scaler = MinMaxScaler()
st.sidebar.header('Section')
section = st.sidebar.radio("Choose a section:", 
                              ("Follower Growth", "Top Hashtags", "Top Mentions", 
                               "Top Influencer","Most Active Time", "Content Performance", 'Network Analysis')
                              )

if section == "Follower Growth":
    
    st.title('Follower Growth of brands in the past two weeks')

    df = pd.read_csv('output/follower_Adidas.csv')
    df2 = pd.read_csv('output/follower_UnderArmour.csv')
    df['Difference'] = df['Follower'].diff()
    df = df.fillna(0)
    df = df.drop(columns=['Unnamed: 0'])
    df2['Difference'] = df2['Follower'].diff()
    df2 = df2.fillna(0)
    df2 = df2.drop(columns=['Unnamed: 0'])
    df2 = df2.rename(columns={'Follower': 'Follower2', 'Difference': 'Difference2'})
    df = df.merge(df2, how='inner', on='Date')
    df['Changes'] =  df.loc[1:, 'Follower'] - df.at[0, 'Follower']
    df['Changes2'] =  df.loc[1:, 'Follower2'] - df.at[0, 'Follower2']
    df = df.fillna(0)

    fig1 = plt.figure(figsize=(35,15)) 
    plt.plot(df['Date'], df['Changes'], 'ro-', label = 'Adidas')
    plt.plot(df['Date'], df['Changes2'], 'bo-', label = 'UnderArmour')
    plt.xticks(rotation='horizontal', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('Follower Growth of brands in the past two weeks', fontsize = 40)
    plt.xlabel('Date', fontsize = 30)
    plt.ylabel('Follower Growth', fontsize = 30)
    plt.legend(fontsize = 25)
    
    for w, x, y, z, a, b in zip(df['Date'], df['Difference'], df['Date'], df['Difference2'], df['Changes'], df['Changes2']):
        
        if x > 0:
            label = "+" + str(int(x))
        else:
            label = str(int(x))
    
        if z > 0:
            label2 = "+" + str(int(z))
        else:
            label2 = str(int(z))

        plt.annotate(label, (w, a), textcoords = "offset points", xytext = (0,10), ha = 'center', fontsize = 25)
        plt.annotate(label2, (y, b), textcoords = "offset points", xytext = (0,10), ha = 'center', fontsize = 25)

    st.pyplot(fig1)
    st.write('Changes in the number of followers of Adidas:', int(df['Difference'].sum()))
    st.write('Changes in the number of followers of UnderArmour:', int(df['Difference2'].sum()))
    
elif section == "Top Hashtags":
    
    st.title('Top Hashtags among brands')
    
    def get_hashtags(tweet):
        entities = tweet.get('entities', {})
        hashtags = entities.get('hashtags', [])
        return [tag['text'].lower() for tag in hashtags]

    t = []
    co = []
    t2 = []
    co2 = []
    
    
    def getHashes(screen_name):
        if __name__ == '__main__':    
            #Jan-Feb Data
            #fname = "output/user_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #fname = "output/user_home_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #One year Data 
            fname = "output/user_timeline_{}_between_2020-2021.json".format(screen_name)
            with open(fname, 'r') as f:
                hashtags = Counter()
                for line in f:
                    tweet = json.loads(line)
                    hashtags_in_tweet = get_hashtags(tweet)
                    hashtags.update(hashtags_in_tweet)
                    for tag, count in hashtags.most_common(10):
                        t.append(tag)
                        co.append(count)
                        #print("{}: {}".format(tag, count))
                
            d = {'Hashtag': t, 'Count': co}
            df = pd.DataFrame(data = d)
            df = df.drop_duplicates()
            df = df.nlargest(38,['Count'])
            fig2 = plt.figure(figsize=(20,8))
            plt.barh(df['Hashtag'], df['Count'], label = "Hashtag", color = 'green')
            plt.xlabel('Hashtag', fontsize = 30)
            plt.ylabel('Count', fontsize = 30)
            plt.title('Most popular hashtag of {}'.format(screen_name), fontsize = 35)
            plt.xticks(rotation = 'horizontal', fontsize = 20)
            plt.yticks(fontsize = 20)
            
            st.pyplot(fig2)
            
    def getHashes2(screen_name):
        if __name__ == '__main__':    
            #Jan-Feb Data
            #fname = "output/user_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #fname = "output/user_home_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #One year Data 
            fname = "output/user_timeline_{}_between_2020-2021.json".format(screen_name)
            with open(fname, 'r') as f:
                hashtags = Counter()
                for line in f:
                    tweet = json.loads(line)
                    hashtags_in_tweet = get_hashtags(tweet)
                    hashtags.update(hashtags_in_tweet)
                    for tag, count in hashtags.most_common(10):
                        t2.append(tag)
                        co2.append(count)
                        #print("{}: {}".format(tag, count))
                        
        d = {'Hashtag': t2, 'Count': co2}
        df = pd.DataFrame(data = d)
        df = df.drop_duplicates()
        df = df.nlargest(33,['Count'])
        fig2 = plt.figure(figsize=(20,8))
        plt.barh(df['Hashtag'], df['Count'], label = "Hashtag", color = 'red')
        plt.xlabel('Hashtag', fontsize = 30)
        plt.ylabel('Count', fontsize = 30)
        plt.title('Most popular hashtag of {}'.format(screen_name), fontsize = 35)
        plt.xticks(rotation = 'horizontal', fontsize = 20)
        plt.yticks(fontsize = 20)
        
        st.pyplot(fig2)
        
    getHashes('Adidas')
    getHashes2('UnderArmour')
    
elif section == 'Top Mentions':
    
    st.title('Top User Mentions by brands')
    
    def get_mentions(tweet):
        entities = tweet.get('entities', {})
        mentions = entities.get('user_mentions', [])
        return [tag['screen_name'] for tag in mentions]
    
    us = []
    co = []
    us2 = []
    co2 = []
    
    def getMentiones(screen_name):
        if __name__ == '__main__':
            #fname = "output/user_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #fname = "output/user_home_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #One year data 
            fname = "output/user_timeline_{}_between_2020-2021.json".format(screen_name)
            
            with open(fname, 'r') as f:
                users = Counter()
                for line in f:
                    tweet = json.loads(line)
                    mentions_in_tweet = get_mentions(tweet)
                    users.update(mentions_in_tweet)
                    for user, count in users.most_common(10):
                        us.append(user)
                        co.append(count)
                        #print("{}: {}".format(user, count))
                        
        d = {'Mention': us, 'Count': co}
        df = pd.DataFrame(data = d)
        df = df.drop_duplicates()
        df = df[df.Mention != 'adidas']
        df = df.nlargest(36,['Count'])
        fig3 = plt.figure(figsize=(20,8))
        plt.barh(df['Mention'], df['Count'], label = "Mention", color = 'yellow')
        plt.xlabel('Mention', fontsize = 30)
        plt.ylabel('Count', fontsize = 30)
        plt.title('Most popular user mentioned by {}'.format(screen_name), fontsize = 35)
        plt.xticks(rotation = 'horizontal', fontsize = 20)
        plt.yticks(fontsize = 20)
        
                        
        st.pyplot(fig3)
        
    def getMentiones2(screen_name):
        if __name__ == '__main__':
            #fname = "output/user_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #fname = "output/user_home_timeline_{}_between_1JanTo1Feb.json".format(screen_name)
            #One year data 
            fname = "output/user_timeline_{}_between_2020-2021.json".format(screen_name)
            
            with open(fname, 'r') as f:
                users = Counter()
                for line in f:
                    tweet = json.loads(line)
                    mentions_in_tweet = get_mentions(tweet)
                    users.update(mentions_in_tweet)
                    for user, count in users.most_common(10):
                        us2.append(user)
                        co2.append(count)
                        #print("{}: {}".format(user, count))
                        
        d = {'Mention': us2, 'Count': co2}
        df = pd.DataFrame(data = d)
        df = df[df.Mention != 'UnderArmour']
        df = df.drop_duplicates()
        df = df.nlargest(25,['Count'])
        fig3 = plt.figure(figsize=(20,8))
        plt.barh(df['Mention'], df['Count'], label = "Mention", color = 'magenta')
        plt.xlabel('Mention', fontsize = 30)
        plt.ylabel('Count', fontsize = 30)
        plt.title('Most popular user mentioned by {}'.format(screen_name), fontsize = 35)
        plt.xticks(rotation = 'horizontal', fontsize = 20)
        plt.yticks(fontsize = 20)
        
                        
        st.pyplot(fig3)
                        
    getMentiones("Adidas")
    getMentiones2("UnderArmour")
    
elif section == "Top Influencer":
    
    st.title('Top Influencer among brands')
    
    #After collecting the 14days of who @Adidas, we start to do the metric of Top Influencer here. 
    def get_mentions(tweet):
        entities = tweet.get('entities', {})
        mentions = entities.get('user_mentions', [])
        return [tag['screen_name'] for tag in mentions]
    
    us = []
    co = []
    us2 = []
    co2 = []
    
    def getMentiones(screen_name):
        if __name__ == '__main__':
            #for i in range(18,25):
            #fname = fdir + 'searchQuery_mgag_before'+'_'+date+'.json'
            fname = "output/searchQuery_Adidas.json"
            with open(fname, 'r') as f:
                users = Counter()
                for line in f:
                    tweet = json.loads(line)
                    mentions_in_tweet = get_mentions(tweet)
                    users.update(mentions_in_tweet)
                    for user, count in users.most_common(20):
                        us.append(user)
                        co.append(count)
                        #print("{}: {}".format(user, count))
                        
        d = {'User': us, 'Count': co}
        df = pd.DataFrame(data = d)
        df = df.drop(df.index[0])
        df = df[df.User != 'adidas']
        df = df.drop_duplicates()
        df = df.nlargest(850,['Count'])
        fig4 = plt.figure(figsize=(30,20)) 
        plt.bar(df['User'], df['Count'], label = "Mentions", color = 'brown')
        plt.xlabel('User', fontsize = 30)
        plt.ylabel('Count', fontsize = 30)
        plt.title('Top influencer of {} '.format(screen_name), fontsize = 35)
        plt.xticks(rotation = 'vertical', fontsize=20)
        plt.yticks(fontsize = 20)
        
        st.pyplot(fig4)
        
    def getMentiones2(screen_name):
        if __name__ == '__main__':
            #for i in range(18,25):
            #fname = fdir + 'searchQuery_mgag_before'+'_'+date+'.json'
            fname = "output/searchQuery_UnderArmour.json"
            with open(fname, 'r') as f:
                users = Counter()
                for line in f:
                    tweet = json.loads(line)
                    mentions_in_tweet = get_mentions(tweet)
                    users.update(mentions_in_tweet)
                    for user, count in users.most_common(20):
                        us2.append(user)
                        co2.append(count)
                        #print("{}: {}".format(user, count))
                        
        d = {'User': us2, 'Count': co2}
        df = pd.DataFrame(data = d)
        df = df.drop(df.index[0])
        df = df[df.User != 'UnderArmour']
        df = df.drop_duplicates()
        df = df.nlargest(3000,['Count'])
        fig5 = plt.figure(figsize=(30,20)) 
        plt.bar(df['User'], df['Count'], label = "Mentions", color = 'purple')
        plt.xlabel('User', fontsize = 30)
        plt.ylabel('Count', fontsize = 30)
        plt.title('Top influencer of {} '.format(screen_name), fontsize = 35)
        plt.xticks(rotation = 'vertical', fontsize=20)
        plt.yticks(fontsize = 20)
        
        st.pyplot(fig5)

                
    getMentiones("Adidas")
    getMentiones2("UnderArmour")
    
elif section == "Most Active Time":
    
    st.title("Most Active Months among brands")
    
    def getActiveHomeTimeLine(period, screen_name):
    #One month data
    #file = 'output/user_home_timeline_{}_between_1JanTo1Feb.json'.format(screen_name)
    #file = 'output/user_timeline_{}_between_1JanTo1Feb.json'.format(screen_name)
    #One year data 
        file = 'output/user_timeline_{}_between_2020-2021.json'.format(screen_name)
        dic = {"date":[],"favorite_count":[], "retweet_count":[]}
        with open(file) as f:
            for line in f:
                tweet = json.loads(line)
                favCount = tweet['favorite_count']
                retCount = tweet['retweet_count']
                d = tweet['created_at'].split(' ')
                #print(d)
                if period=='daily':
                    d = d[2]+d[1]+d[5]
                elif period=='monthly':
                    d = d[1]+d[5]
                elif period=='yearly':
                    d = d[5]
                if d not in dic['date']:
                    dic['date'].append(d)
                    dic['favorite_count'].append(favCount)
                    dic['retweet_count'].append(retCount)
                else:
                    idx = dic['date'].index(d)
                    dic['favorite_count'][idx] += favCount
                    dic['retweet_count'][idx] += retCount
                    
        df = pd.DataFrame(dic)
        #df[["favorite_count", "retweet_count"]] = min_max_scaler.fit_transform(df[["favorite_count","retweet_count"]])
        df = df.iloc[::-1]
        fig6 = plt.figure(figsize=(20,8)) 
        plt.plot(df['date'], df['favorite_count'], label = "Favorite", color = 'orange')
        plt.bar(df['date'], df['retweet_count'], label = "Retweet", color = 'blue')
        plt.legend()
        plt.xlabel('Month', fontsize = 25)
        plt.ylabel('Count', fontsize = 25)
        plt.title('Most Popular Month of {}'.format(screen_name), fontsize = 30)
        plt.xticks(rotation = 'horizontal', fontsize = 20)
        plt.yticks(fontsize = 20)
        
        st.pyplot(fig6)

    getActiveHomeTimeLine('monthly',"Adidas")
    getActiveHomeTimeLine('monthly',"UnderArmour")
    
elif section == 'Content Performance':
    
    st.title("Comparison of content performance among brands")
    
    #This is to get the January Text from the Adidas "User Timeline" and to conduct the analysis.
#This is used to downloaded dataset from the tweets collect functions
#Not the HOME TIMELINE OF ADIDAS!!
    def getUserTimelineTweetPerformance(screen_name, screen_name2):
        #file = 'output/user_timeline_{}_between_1JanTo1Feb.json'.format(screen_name)
        file = 'output/user_timeline_{}_between_2020-2021.json'.format(screen_name)
        file2 = 'output/user_timeline_{}_between_2020-2021.json'.format(screen_name2)
        dic = {"date":[],"favorite_count":[], "retweet_count":[]}
        dic2 = {"date":[],"favorite_count":[], "retweet_count":[]}
        count=0
        count2=0
        with open(file) as f:
            for line in f:
                tweet = json.loads(line)
                favCount = tweet['favorite_count']
                retCount = tweet['retweet_count']
                d = tweet['created_at'].split(' ')
                dic['date'].append(d)
                dic['favorite_count'].append(favCount)
                dic['retweet_count'].append(retCount)
                count += 1
                
        with open(file2) as f:
            for line in f:
                tweet = json.loads(line)
                favCount = tweet['favorite_count']
                retCount = tweet['retweet_count']
                d = tweet['created_at'].split(' ')
                dic2['date'].append(d)
                dic2['favorite_count'].append(favCount)
                dic2['retweet_count'].append(retCount)
                count2 += 1
        
        df = pd.read_csv('output/follower_Adidas.csv')
        followersCount = df['Follower'].iloc[-1]
        df2 = pd.read_csv('output/follower_UnderArmour.csv')
        followersCount2 = df2['Follower'].iloc[-1]
        #print("Followers Count : ",followersCount)
        dfDaily = pd.DataFrame(dic)
        dfDaily2 = pd.DataFrame(dic2)
        #st.write("Statuses Count : ",count,count2)
        favouritesCount = dfDaily['favorite_count'].sum()
        favouritesCount2 = dfDaily2['favorite_count'].sum()
        #st.write("Favourites Count : ",favouritesCount, favouritesCount2)
        retweetsCount = sum(dfDaily['retweet_count'])
        retweetsCount2 = sum(dfDaily2['retweet_count'])
        #st.write("Retweets Count : ",retweetsCount,retweetsCount2)
        avgFavouritesCount = round(favouritesCount / count, 4)
        avgFavouritesCount2 = round(favouritesCount2 / count2, 4)
        #st.write("Average Favourite Count : ",avgFavouritesCount,avgFavouritesCount2)
        avgRetweetsCount = round(retweetsCount / count, 4)
        avgRetweetsCount2 = round(retweetsCount2 / count, 4)
        #st.write("Average Tweets Count : ",avgRetweetsCount,avgRetweetsCount2)

        favoritPerUser = round(favouritesCount / followersCount, 4)
        favoritPerUser2 = round(favouritesCount2 / followersCount2, 4)
        
        retweetPerUser = round(retweetsCount / followersCount, 4)
        retweetPerUser2 = round(retweetsCount2 / followersCount2, 4)
        
        df3 = pd.DataFrame([[followersCount,count,favouritesCount,retweetsCount,avgFavouritesCount,avgRetweetsCount
                             ,favoritPerUser,retweetPerUser],
                        [followersCount2,count2,favouritesCount2,retweetsCount2,avgFavouritesCount2,avgRetweetsCount2
                         ,favoritPerUser2,retweetPerUser2]],
                       columns=['Followers Count','Statuses Count','Favourites Count','Retweets Count','Average Favourite Count', 
                                'Average Retweets Count', 'Favorite Per User', 'Retweet Per User'],
                       index=[screen_name, screen_name2])
        
        df3 = df3.transpose()
        
        df3['Winner'] = ''
        df3.loc[df3.Adidas > df3.UnderArmour, 'Winner'] = 'Adidas'
        df3.loc[df3.UnderArmour > df3.Adidas, 'Winner'] = 'UnderArmour'
        
        st.table(df3.astype('object'))
        
        categories = ['Followers Count','Statuses Count', 'Favourites Count', 'Retweets Count',
                      'Average Favourite Count', 'Average Retweets Count', 
                      'Favorite Per User', 'Retweet Per User']
        categories = [*categories, categories[0]]
        
        fc = [followersCount,followersCount2]
        norm1 = np.linalg.norm(fc)
        followersCount = followersCount/norm1
        followersCount2 = followersCount2/norm1
        
        sc = [count,count2]
        norm5 = np.linalg.norm(sc)
        count = count/norm5
        count2 = count2/norm5
        
        fcc = [favouritesCount,favouritesCount2]
        norm6 = np.linalg.norm(fcc)
        favouritesCount = favouritesCount/norm6
        favouritesCount2 = favouritesCount2/norm6
        
        rc = [retweetsCount,retweetsCount2]
        norm7 = np.linalg.norm(rc)
        retweetsCount = retweetsCount/norm7
        retweetsCount2 = retweetsCount2/norm7
        
        afc = [avgFavouritesCount,avgFavouritesCount2]
        norm2 = np.linalg.norm(afc)
        avgFavouritesCount = avgFavouritesCount/norm2
        avgFavouritesCount2 = avgFavouritesCount2/norm2
        
        arc = [avgRetweetsCount,avgRetweetsCount2]
        norm3 = np.linalg.norm(arc)
        avgRetweetsCount = avgRetweetsCount/norm3
        avgRetweetsCount2 = avgRetweetsCount2/norm3
        
        fpu = [favoritPerUser,favoritPerUser2]
        norm4 = np.linalg.norm(fpu)
        favoritPerUser = favoritPerUser/norm4
        favoritPerUser2 = favoritPerUser2/norm4
        
        rpu = [retweetPerUser,retweetPerUser]
        norm5 = np.linalg.norm(rpu)
        retweetPerUser = retweetPerUser/norm5
        retweetPerUser2 = retweetPerUser2/norm5
        
        
        Adidas = [followersCount,count,favouritesCount,retweetsCount,avgFavouritesCount,avgRetweetsCount
                             ,favoritPerUser,retweetPerUser]
        
        UnderArmour = [followersCount2,count2,favouritesCount2,retweetsCount2,avgFavouritesCount2,avgRetweetsCount2
                             ,favoritPerUser2,retweetPerUser2]
        
        Adidas = [*Adidas, Adidas[0]]
        UnderArmour = [*UnderArmour, UnderArmour[0]]
        
        label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(Adidas))
        
        fig8 = plt.figure(figsize=(6, 6))
        plt.subplot(polar=True)
        plt.plot(label_loc, Adidas, label='Adidas', color = 'red')
        plt.plot(label_loc, UnderArmour, label='UnderArmour', color = 'blue')
        plt.title('Brand comparison', size = 15)
        lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
        plt.yticks(fontsize = 8)
        plt.xticks(fontsize = 8)
        plt.tick_params(axis='y', labelsize=0, length = 0)
        plt.legend(fontsize = 6)
        
        st.pyplot(fig8)
        
    getUserTimelineTweetPerformance("Adidas", "UnderArmour")
        
elif section == 'Network Analysis':

    select_feature = st.radio('Select a brand:', ('Adidas','UnderArmour'))
    
    if select_feature == 'Adidas':
        
        df = pd.read_csv("ReciprocalFriendAdidas.csv")
        df = df.dropna()
        
        x_point = list(df[df.columns[0]].values)
        y_point = list(df[df.columns[1]].values)
        edges_list = []
        for i in range(len(x_point)):
            edges_list.append((x_point[i], y_point[i]))
            node_list = set(x_point+y_point)
            RG = nx.Graph()
            RG.add_nodes_from(node_list)
            RG.add_edges_from(edges_list)
        
        components = nx.connected_components(RG)
        largestSubgraph = max(components, key=len)
        components = nx.connected_components(RG)
        minSubgraph = min(components, key=len)
            
        df2 = pd.DataFrame([[RG.number_of_nodes(),RG.number_of_edges(),nx.density(RG),len(largestSubgraph),
                             len(minSubgraph)]], 
                           columns=['Nodes','Edges', 'Network Density','Length of largest subgraph',
                                    'Length of smallest subgraph'], 
                           index = ['Adidas'])
        
        df2 = df2.transpose()
        
        st.subheader('Network Statistics of Adidas:')
        
        st.table(df2.astype('object'))
        
        person_dict = dict(RG.degree(RG.nodes))
        nx.set_node_attributes(RG, name="degree", values=person_dict)
        
        betweenness_dict = nx.betweenness_centrality(RG)
        eigen_dict = nx.eigenvector_centrality(RG, max_iter=200)
        closeness_dict = nx.closeness_centrality(RG)
        
        nx.set_node_attributes(RG, name="betweenness", values="betweenness_dict")
        nx.set_node_attributes(RG, name="eigenvector", values="eigen_dict")
        nx.set_node_attributes(RG, name="closenessvector", values="closeness_dict")

        sortedBetweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
        sortedEigen = sorted(eigen_dict.items(), key=itemgetter(1), reverse=True)
        sortedCloseness = sorted(closeness_dict.items(), key=itemgetter(1), reverse=True)
        topBetweenness = sortedBetweenness[:10]
        topEigen = sortedEigen[:10]
        topCloseness = sortedCloseness[:10]

        #Then find and print their degree
        nm = []
        de = []
        b = []
        
        for p in topBetweenness:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            b.append(p[1])
            
        nm = map(int, nm)
            
        d = {'Name': nm, 'Degree': de, 'Betweenness': b}
        df3 = pd.DataFrame(data = d)
        #df3['Name'] = df3['Name'].astype(int)
            
        st.subheader("Top 10 nodes by betweenness centrality:")
        
        st.table(df3.astype('object'))
        
        nm = []
        de = []
        e = []
        
        for p in topEigen:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            e.append(p[1])
            
        nm = map(int, nm)
            
        d = {'Name': nm, 'Degree': de, 'Eigen': e}
        df4 = pd.DataFrame(data = d)
        #df4['Name'] = df4['Name'].astype(int)
            
        st.subheader("Top 10 nodes by eigen centrality:")
        
        st.table(df4.astype('object'))
        
        nm = []
        de = []
        c = []
        
        for p in topCloseness:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            c.append(p[1])
            
        nm = map(int, nm)
            
        d = {'Name': nm, 'Degree': de, 'Closeness': c}
        df5 = pd.DataFrame(data = d)
        #df5['Name'] = df5['Name'].astype(int)
            
        st.subheader("Top 10 nodes by closeness centrality:")
        
        st.table(df5.astype('object'))
        
        st.subheader('Network Graph of Adidas:')
        
        st.image('Adidas Network.png')
        
        partition = community.best_partition(RG)
        
        st.subheader('Modularity of Adidas:')
        
        modularity = {} # Create a new, empty dictionary
        for k,v in partition.items(): # Loop through the community dictionary
            if v not in modularity:
                modularity[v] = [k] # Add a new key for a modularity class the code hasn't seen before
            else:
                modularity[v].append(k) # Append a name to the list for a modularity class the code has already seen
                    
        for k,v in modularity.items(): # Loop through the new dictionary
            st.write('Class '+str(k)+':', v) # Print out the classes and their members
        
        #fig = plt.figure(figsize=(120,120))
        #plt.title('Network Graph of Adidas')
        #nx.draw_networkx(RG, pos=nx.spring_layout(RG))
        #st.pyplot(fig)
                    
    elif select_feature == 'UnderArmour':
        
        df = pd.read_csv("ReciprocalFriendUnderArmour.csv")
        df = df.dropna()
        
        x_point = list(df[df.columns[0]].values)
        y_point = list(df[df.columns[1]].values)
        edges_list = []
        for i in range(len(x_point)):
            edges_list.append((x_point[i], y_point[i]))
            node_list = set(x_point+y_point)
            RG = nx.Graph()
            RG.add_nodes_from(node_list)
            RG.add_edges_from(edges_list)
            
        components = nx.connected_components(RG)
        largestSubgraph = max(components, key=len)
        components = nx.connected_components(RG)
        minSubgraph = min(components, key=len)
 
        df2 = pd.DataFrame([[RG.number_of_nodes(),RG.number_of_edges(),nx.density(RG),len(largestSubgraph),
                             len(minSubgraph)]], 
                           columns=['Nodes','Edges', 'Network Density','Length of largest subgraph',
                                    'Length of smallest subgraph'], 
                           index = ['UnderArmour'])
        
        df2 = df2.transpose()
        
        st.subheader('Network Statistics of UnderArmour:')
        
        st.table(df2.astype('object'))
        
        person_dict = dict(RG.degree(RG.nodes))
        nx.set_node_attributes(RG, name="degree", values=person_dict)
        
        betweenness_dict = nx.betweenness_centrality(RG)
        eigen_dict = nx.eigenvector_centrality(RG, max_iter=200)
        closeness_dict = nx.closeness_centrality(RG)
        
        nx.set_node_attributes(RG, name="betweenness", values="betweenness_dict")
        nx.set_node_attributes(RG, name="eigenvector", values="eigen_dict")
        nx.set_node_attributes(RG, name="closenessvector", values="closeness_dict")

        sortedBetweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
        sortedEigen = sorted(eigen_dict.items(), key=itemgetter(1), reverse=True)
        sortedCloseness = sorted(closeness_dict.items(), key=itemgetter(1), reverse=True)
        topBetweenness = sortedBetweenness[:10]
        topEigen = sortedEigen[:10]
        topCloseness = sortedCloseness[:10]

        #Then find and print their degree
        
        nm = []
        de = []
        b = []
        
        for p in topBetweenness:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            b.append(p[1])
            
        nm = map(int, nm)
            
        d = {'Name': nm, 'Degree': de, 'Betweenness': b}
        df3 = pd.DataFrame(data = d)
        #df3['Name'] = df3['Name'].astype(int)
            
        st.subheader("Top 10 nodes by betweenness centrality:")
        
        st.table(df3.astype('object'))
        
        nm = []
        de = []
        e = []
        
        for p in topEigen:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            e.append(p[1])
        
        nm = map(int, nm)
        
        d = {'Name': nm, 'Degree': de, 'Eigen': e}
        df4 = pd.DataFrame(data = d)
        #df4['Name'] = df4['Name'].astype(int)
            
        st.subheader("Top 10 nodes by eigen centrality:")
        
        st.table(df4.astype('object'))
        
        nm = []
        de = []
        c = []
        
        for p in topCloseness:
            
            nm.append(p[0])
            de.append(person_dict[p[0]])
            c.append(p[1])
            
        nm = map(int, nm)
            
        d = {'Name': nm, 'Degree': de, 'Closeness': c}
        df5 = pd.DataFrame(data = d)
        #df5['Name'] = df5['Name'].astype(int)
            
        st.subheader("Top 10 nodes by closeness centrality:")
        
        st.table(df5.astype('object'))
        
        st.subheader('Network Graph of UnderArmour:')
        
        st.image('UnderArmour Network.png')
        
        partition = community.best_partition(RG)
        
        st.subheader('Modularity of UnderArmour:')
        
        modularity = {} # Create a new, empty dictionary
        for k,v in partition.items(): # Loop through the community dictionary
            if v not in modularity:
                modularity[v] = [k] # Add a new key for a modularity class the code hasn't seen before
            else:
                modularity[v].append(k) # Append a name to the list for a modularity class the code has already seen
                    
        for k,v in modularity.items(): # Loop through the new dictionary
            st.write('Class '+str(k)+':', v) # Print out the classes and their members
        
        #fig = plt.figure(figsize=(120,120))
        #plt.title('Network Graph of UnderArmour')
        #nx.draw_networkx(RG, pos=nx.spring_layout(RG))
        #st.pyplot(fig)
        
# =============================================================================
#         st.write("\n------------------------- Brands statistics based on Favourties and Retweets -------------------------")
#         st.write("Favourited {} times ({} per tweet, {} per user) of {}"
#                  .format(favouritesCount, avgFavouritesCount, favoritPerUser, screen_name))
#         st.write("Favourited {} times ({} per tweet, {} per user) of {}"
#                  .format(favouritesCount2, avgFavouritesCount2, favoritPerUser2, screen_name2))
#         st.write("Retweeted {} times ({} per tweet, {} per user of {})"
#                  .format(retweetsCount, avgRetweetsCount, retweetPerUser, screen_name))
#         st.write("Retweeted {} times ({} per tweet, {} per user of {})"
#                  .format(retweetsCount2, avgRetweetsCount2, retweetPerUser2, screen_name2))
# =============================================================================
    
        #return pd.DataFrame(dic)