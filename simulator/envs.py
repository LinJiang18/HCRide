import pandas as pd
import numpy as np
import statistics
from simulator.regions import *
from simulator.orders import *
from simulator.unitity import *
from simulator.drivers import *
import random

import math
from scipy.stats import expon

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
plt.rcParams["font.family"] = "Times New Roman"

#rewardList = []
#gamma = 0.98

class Env(object):
    def __init__(self,driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime):
        self.driverPreInit = driverPreInit
        self.driverLocInit = driverLocInit
        self.orderInfo = orderInfo
        self.locRange = locRange # locRange = [minlon,maxlon,minlat,maxlat]
        self.length = M
        self.width = N
        self.maxTime = maxTime
        self.maxDay = maxDay


        self.cityTime = 0 #
        self.cityDay = 0
        self.maxCityTime = maxTime


        self.driverList = []
        self.driverDict = {}
        self.driverNum = driverNum



        self.M = M
        self.N = N
        self.regionNum = self.M * self.N
        self.regionList = [Region(i,self.regionNum) for i in range(self.regionNum)]  # 区域的结点列表
        self.regionDict = {}

        self.candidateDriverSize = 20
        self.maxDriverPreNum = 0

        # speed
        self.speed = 30

        #
        self.alpha = 1.5
        self.gamma = 0.98

        self.globalwtList = []


    def set_region_info(self):
        for i in range(self.regionNum):
            region = self.regionList[i]
            self.regionDict[i] = region
            region.set_neighbors()
            region.set_region_meanwt()

    def set_driver_info(self, driverPreInit, driverLocInit):
        for i in range(self.driverNum):
            driverID = i
            driverPre = driverPreInit[i]
            driverRegion = driverLocInit[i]
            driverLoc = generate_loc(driverRegion)
            driver = Driver(driverID, driverPre, driverLoc)
            self.driverList.append(driver)
            self.driverDict[driverID] = driver

    def set_day_info(self, dayIndex):
        self.cityDay = dayIndex
        for driver in self.driverList:
            driver.set_day_info(dayIndex)
        for region in self.regionList:
            region.set_day_info(dayIndex)

    def reset_clean(self):
        self.cityTime = 0
        self.dayOrder = []
        for driver in self.driverList:
            driver.reset_driver_info()
        for region in self.regionList:
            region.reset_region_info()
        self.boost_one_day_order()
        self.boost_step_order_info(self.cityTime)
        self.boost_step_region_info()

    def boost_step_order_info(self, T):
        stepOrderList = self.dayOrder[T]
        for order in stepOrderList:
            region = order.oriRegion
            self.regionList[region].add_order(order)


    def boost_step_region_info(self):
        for driver in self.driverList:
            if driver.state == 1:
                region = driver.region
                self.regionList[region].add_driver(driver)

    def boost_one_day_order(self):
        dayOrderList = [[] for _ in np.arange(self.maxCityTime)]
        for dayOrder in self.orderInfo[self.cityDay]:
            for order in dayOrder:
                startTime = order[2]
                # orderID,orderDay,orderMin,orderRegion,oriLon,oriLat,destLon,destLat,price
                orderRegion = self.regionList[order[3]]
                dayOrderList[startTime].append(Order(order[0], order[1], order[2], orderRegion, order[4], order[5],
                                                     order[7], order[8], order[9]))
        self.dayOrder = dayOrderList


    def add_global_wt(self,wt):
        self.globalwtList.append(wt)

    def cal_global_mean_wt(self):
        return float(np.mean(np.array(self.globalwtList)))

    def cal_global_max_wt(self):
        regionMeanWTList = []
        for region in self.regionList:
            regionMeanWTList.append(region.meanwt)
        MaxWT = max(regionMeanWTList)
        return MaxWT



    def driver_collect(self,order):
        orderRegion = order.orderRegion
        neighborLevelIndex = 3
        driverList = []
        neighborList = orderRegion.neighborLevel[neighborLevelIndex]
        for neighbor in neighborList:
            driverList.append(self.regionList[neighbor].driverList)
        driverList = [x for y in driverList for x in y]

        while len(driverList) == 0:
            neighborLevelIndex += 1
            if neighborLevelIndex == 4:
                return 0
            driverList = []
            neighborList = orderRegion.neighborLevel[neighborLevelIndex]
            for neighbor in neighborList:
                driverList.append(self.regionList[neighbor].driverList)
            driverList = [x for y in driverList for x in y]

        return driverList

    def generate_candidate_set(self,order,driverList):
        disList = []
        for i in range(len(driverList)):
            driver = driverList[i]
            dis = cal_dis(order.oriLoc,driver.loc)
            disList.append((dis,i))
        disList = sorted(disList,key = lambda x:x[0],reverse = False)
        disList = disList[:self.candidateDriverSize]
        disList = [x[1] for x in disList]
        candidateList = [driverList[x] for x in disList]

        return candidateList

    def driver_state_calculate(self,driverList):
        driverArray = np.zeros((len(driverList),self.maxDriverPreNum + 6))
        index = 0
        for driver in driverList:
            region = np.array([driver.region + 1]) / self.regionNum # 1
            regionWT = self.regionList[driver.region].meanwt
            regionWT = np.array([regionWT])
            t = np.array([driver.cityTime]) / self.maxCityTime
            lon = (driver.loc.lon - minlon) / (maxlon - minlon)
            lon = np.array([lon])
            lat = (driver.loc.lat - minlat) / (maxlat - minlat)
            lat = np.array([lat])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderDis = cal_dis(driver.loc,order.oriLoc)
                if orderDis <= nearwt:
                    nearwt = orderDis
            nearwt = np.array([nearwt])
            preRegionList = driver.preRegion
            preRegionList = [(x+1)/self.regionNum for x in preRegionList]
            driverState = np.concatenate((region,regionWT,lon,lat,nearwt,t))
            driverArray[index,:] = driverState
            index += 1
        return driverArray

    # def driver_next_state_calculate(self,driver,order,trs):
    #     region = np.array([order.destRegion + 1])
    #     t = np.array([driver.cityTime + trs])
    #     lon = round((order.destLoc.lon - minlon) / (maxlon - minlon), 5)
    #     lon = np.array([lon])
    #     lat = round((order.destLoc.lon - minlat) / (maxlat - minlat), 5)
    #     lat = np.array([lat])
    #     preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')
    #     nextDriverState = np.concatenate((region,t,lon,lat,preRegion))
    #     return nextDriverState

    def action_state_calculate(self,driverList,order):
        actionArray = np.zeros((len(driverList),10))
        oriRegion = np.array([order.oriRegion + 1]) / self.regionNum
        destRegion = np.array([order.destRegion + 1]) / self.regionNum
        oriLon = (order.oriLoc.lon - minlon) / (maxlon - minlon)
        oriLon = np.array([oriLon])
        oriLat = (order.oriLoc.lat - minlat) / (maxlat - minlat)
        oriLat = np.array([oriLat])
        destLon = (order.destLoc.lon - minlon) / (maxlon - minlon)
        destLon = np.array([destLon])
        destLat = (order.destLoc.lat - minlat) / (maxlat - minlat)
        destLat = np.array([destLat])
        oriWT = self.regionList[order.oriRegion].meanwt
        oriWT = np.array([oriWT])
        destWT = self.regionList[order.destRegion].meanwt
        destWT = np.array([destWT])
        dt = np.array([(cal_dis(order.oriLoc, order.destLoc) / self.speed) * 60])
        index = 0
        for driver in driverList:
            wt = np.array([(cal_dis(driver.loc,order.oriLoc) / self.speed) * 60])
            orderState = np.concatenate((oriRegion, destRegion, oriLon, oriLat, destLon, destLat, oriWT, destWT, dt, wt))
            actionArray[index,:] = orderState
            index += 1
        return actionArray

    def con_state_calcualte(self):
        supplyDemandList = []
        for region in self.regionList:
            supply = len(region.driverList)
            demand = len(region.orderList)
            supplyDemandList.append(supply)
            supplyDemandList.append(demand)
        return np.array(supplyDemandList)

    def cal_reward(self,wt,meanwt,trs,cost):
        reward = (-wt) + (self.alpha * (-1) * abs(wt - meanwt) / 3)
        a = (1 - self.alpha) * (-wt)
        b = self.alpha * (-1) * abs(wt - meanwt)
        return reward
        # rewardList.append(reward)
        # if len(rewardList) == 1:
        #     reward = 0
        # else:
        #     reward = reward/statistics.stdev(rewardList)
         #   rt = reward / trs
         #   gammaReward = 0
         #   for t in range(trs):
         #       gammaReward += rt * pow(self.gamma,t)

    def cal_money_reward(self,money):
        return money

    def cal_absolute_reward(self,wt,meanwt,trs,cost):
        reward = ((1 - self.alpha) * (-wt) + self.alpha * (-1) * abs(wt - meanwt))
        return reward


    def cal_maxmin_reward(self,wt,meanwt,trs,cost):
        reward = ((1 - self.alpha) * (-wt) + self.alpha * (-1) * meanwt)
        return reward



    def cal_cost(self,order,driver):
        dest = order.destRegion
        if dest in driver.preRegion:
            cost = 1
        else:
            cost = 0
        return cost

    def cal_money(self,wt,money):
        lambda_ = 7
        wt = (wt - 5) / 60
        pro = expon.cdf(wt, scale=1 / lambda_)
        num = random.random()
        if num < pro:
            money1 = 0
            symbol = 1
        else:
            money1 = money
            symbol = 0
        return money1,symbol






    def step(self,dDict,replayBuffer):
        updateDriverList = []
        for driver in self.driverList:
            symbol = driver.step_update_driver_info()
            if symbol == 1:
                updateDriverList.append(driver)
        for region in self.regionList:
            region.step_update_region_info()
        if self.cityTime < self.maxCityTime - 1:
            self.boost_step_order_info(self.cityTime + 1)
            self.boost_step_region_info()  # update supply and demand
        for driver in updateDriverList:
            region = np.array([driver.region + 1]) / self.regionNum  # 1
            regionWT = self.regionList[driver.region].meanwt
            regionWT = np.array([regionWT])
            t = np.array([driver.cityTime]) / self.maxCityTime
            lon = (driver.loc.lon - minlon) / (maxlon - minlon)
            lon = np.array([lon])
            lat = (driver.loc.lat - minlat) / (maxlat - minlat)
            lat = np.array([lat])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderWT = (cal_dis(driver.loc, order.oriLoc) / self.speed) * 60
                if orderWT <= nearwt:
                    nearwt = orderWT
            nearwt = np.array([nearwt])
            preRegionList = driver.preRegion
            preRegionList = [(x + 1) / self.regionNum for x in preRegionList]
          #  preRegion = np.pad(preRegionList, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')  # 62
            driverState = np.concatenate((region,regionWT,lon,lat,nearwt,t))
            # contextualState = self.con_state_calcualte()
            #next_driver_state = np.concatenate((driverState,contextualState))
            next_driver_state = driverState
            dDict[driver].add_nextState(next_driver_state)
            replayBuffer.add(dDict[driver].matchingState, dDict[driver].state, dDict[driver].action,
                             dDict[driver].reward, dDict[driver].cost,dDict[driver].nextState)
            dDict.pop(driver, None)
        self.cityTime += 1



    def plot(self,y1,y2,y3):


        plt.figure(figsize=(12, 9))
        x = range(1, len(y1) + 1)
        y = [-i for i in y1]
        plt.plot(x, y1, linewidth=3)
        plt.xlabel('Episode', fontsize=50)
        plt.ylabel('Reward', fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 9))
        x = range(1, len(y2) + 1)
        plt.xlim((0, 400))
        plt.ylim((-5, 16))
        plt.plot(x, y2, linewidth=3)
        x1 = np.linspace(-10, 450, 1000)
        y1 = np.zeros_like(x1)
        plt.plot(x1, y1, color='r', linestyle='--', linewidth=3)
        plt.xlabel('Update Iteration', fontsize=50)
        plt.ylabel('Actor Loss', fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16],fontsize=40)
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 9))
        x = range(1, len(y3) + 1)
        plt.plot(x, y3, linewidth=3)
        plt.plot(x1, y1, color='r', linestyle='--', linewidth=3)
        x1 = np.linspace(-10, 450, 1000)
        y1 = np.zeros_like(x1)
        plt.xlim((-10, 400))
        plt.ylim((-50, 1300))
        plt.xlabel('Update Iteration', fontsize=50)
        plt.ylabel('Critic Loss', fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks([0, 200, 400, 600,  800, 1000, 1200], fontsize=40)
        plt.grid()
        plt.tight_layout()
        plt.show()






















