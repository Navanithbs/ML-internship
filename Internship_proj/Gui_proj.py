from tkinter import *
from tkinter import filedialog
import numpy as np
import pandas as pd
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from tkinter import messagebox
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import mean_squared_error





#defining the pop up window
class ABC(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()        


        
        
#assigning the pop up window for file browsing
root = Tk()
app = ABC(master=root)




#Heading of pop up window
app.master.title("Machine Learning")




#Frame forBrowsing file
fb=Frame(app)




#label for file
l=Label(fb,text="Filename:")
l.grid(row=0,column=0)




# text field 
var=StringVar()
text=Entry(fb,textvariable=var)
text.grid(row=0,column=1)





#defining browse function
def browsefunc():
    filename = filedialog.askopenfilename()
    text.delete(0,END)
    text.insert(0,filename)


    
#Adding browse button    
browsebutton = Button(fb, text="Browse", command=browsefunc)
browsebutton.grid(row=0,column=2)




#defining next function
def donefunc():
    f.pack()
    f2.pack()


    
    
#Next button
nextbutton=Button(fb,text="Done",command=donefunc)
nextbutton.grid(row=1,column=2)
fb.pack()




#Frame for Radio button set 1
f=Frame(app)




#label for file
l=Label(f,text="Select technique")
l.grid(row=0,column=0)




#Setting up Radio Button function
vart=IntVar()




#Radio Buttons for learning type 1
R1 = Radiobutton(f, text="Supervised Regression", variable=vart, value=1)
R1.grid( row=1,column=0 )

R2 = Radiobutton(f, text="Supervised Classification", variable=vart, value=2)
R2.grid(row=1,column=1)

R3 = Radiobutton(f, text="Unupervised Learning Clustering", variable=vart, value=3)
R3.grid(row=1,column=3)




#Frame for button
f2=Frame(app)
    
    
    
    
    
    
#function
def selec():
    f3.pack()
    
    # Different models and their score
    if(vart.get()==1):
        supervisedreg.pack()
        unsupervised.pack_forget()
        supervisedclas.pack_forget()
    
    elif(vart.get()==2):
        supervisedreg.pack_forget()
        unsupervised.pack_forget()
        supervisedclas.pack()
        
        
    elif(vart.get()==3):
        supervisedreg.pack_forget()
        unsupervised.pack()
        supervisedclas.pack_forget()




#button
b=Button(f2,text="Selected",command=selec)
b.pack()



#Frame for Radio button set for survised learning
supervisedreg=Frame(app)




#label for file
l=Label(supervisedreg,text="Select Model")
l.grid(row=0,column=0)




#Setting up Radio Button function
vars=IntVar()





#Radio Buttons for Linear Regression
LR = Radiobutton(supervisedreg, text="Linear Regression", variable=vars, value=1)
LR.grid( row=1,column=0 )





#Radio Buttons for Random Forest Regression
RF = Radiobutton(supervisedreg, text="Random Forest", variable=vars, value=2)
RF.grid( row=1,column=1 )





#Radio Buttons for Logistic Regression
LOR = Radiobutton(supervisedreg, text="Logistic Regression", variable=vars, value=3)
LOR.grid( row=1,column=2 )





#Radio Buttons for Support Vector Machine Regression
SVM = Radiobutton(supervisedreg, text="SVM Regression", variable=vars, value=4)
SVM.grid( row=1,column=3 )





#Radio Buttons for Support XGBoost Regression
xg = Radiobutton(supervisedreg, text="XGBoost Regression", variable=vars, value=5 )
xg.grid( row=1,column=4 )






#function for supervised button click
def supervisedreg_button_function():
    filename=var.get()
    if (vars.get()==1):
        result = Linear_regression(filename)
    elif (vars.get()==2):
        result = Decision_Tree_regression(filename)
    elif (vars.get()==3):
        result = Logistic_regression(filename)
    elif (vars.get()==4):
        result = SVM_regression(filename)
    elif (vars.get()==5):
        result = XGB_regression(filename)
    f3.pack()
    textofresult.delete(0,END)
    textofresult.insert(0,result)




#button
supervisedreg_button=Button(supervisedreg,text="Selected",command=supervisedreg_button_function)
supervisedreg_button.grid(row=2,column=0)
    
    
        


#Frame for Radio button set for survised learning
supervisedclas=Frame(app)




#label for file
labelofsupervisedclas=Label(supervisedclas,text="Select Model")
labelofsupervisedclas.grid(row=0,column=0)




#Setting up Radio Button function
varforsupervisedclas=IntVar()





#Radio Buttons for Linear Regression
KM = Radiobutton(supervisedclas, text="KNeighbour", variable=varforsupervisedclas, value=1)
KM.grid( row=1,column=0 )






#Radio button for Decision Trees
DT=Radiobutton(supervisedclas,text="Decision Tree",variable=varforsupervisedclas,value=2)
DT.grid(row=1,column=1)






#Radio button for Naive Bayes
NB=Radiobutton(supervisedclas,text="Naive Bayes",variable=varforsupervisedclas,value=3)
NB.grid(row=1,column=2)






#Radio button for SVM Classifier
SVMC=Radiobutton(supervisedclas,text="SVM",variable=varforsupervisedclas,value=4)
SVMC.grid(row=1,column=3)






#Radio button for XGB Classifier
XGBC=Radiobutton(supervisedclas,text="XGB",variable=varforsupervisedclas,value=5)
XGBC.grid(row=1,column=4)






#function for supervised classification button click
def supervisedclas_button_function():
    filename=var.get()
    if (varforsupervisedclas.get()==1):
        result = KNeighbour(filename)
    elif (varforsupervisedclas.get()==2):
        result=DecisionTreeClas(filename)
    elif (varforsupervisedclas.get()==3):
        result=NaiveBayesClas(filename)
    elif (varforsupervisedclas.get()==4):
        result=SVMClas(filename)
    elif (varforsupervisedclas.get()==5):
        result=XGBClas(filename)
    f3.pack()
    textofresult.delete(0,END)
    textofresult.insert(0,result)

    
    
    
#button
supervisedclas_button=Button(supervisedclas,text="Selected",command=supervisedclas_button_function)
supervisedclas_button.grid(row=2,column=0)

    
  
    
#Frame for Radio button set for survised learning
unsupervised=Frame(app)




#label for file
labelofunsupervised=Label(unsupervised,text="Select Model")
labelofunsupervised.grid(row=0,column=0)




#Setting up Radio Button function
varforunsupervised=IntVar()





#Radio Buttons for Linear Regression
KM = Radiobutton(unsupervised, text="KMeans", variable=varforunsupervised, value=1)
KM.grid( row=1,column=0 )




#function for supervised classification button click
def unsupervised_button_function():
    filename=var.get()
    if (varforunsupervised.get()==1):
        result = KMean(filename)
    f3.pack()
    textofresult.delete(0,END)
    textofresult.insert(0,result)





#button
unsupervised_button=Button(unsupervised,text="Selected",command=unsupervised_button_function)
unsupervised_button.grid(row=2,column=0)

    
    
    
    

# Machine Learning Functions
def Linear_regression(filename):
    model=LinearRegression()
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


    
    
    
    
def Decision_Tree_regression(filename):
    model=RandomForestRegressor()
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""

    
    
    
    

# Machine Learning Functions
def Logistic_regression(filename):
    model=LogisticRegression()
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return "" 

    
    
    
    

# Machine Learning Functions
def SVM_regression(filename):
    answer = simpledialog.askstring("SVM", "Kernel Type(linear/poly/rbf/sigmoid)",
                                parent=app)
    try:
        model=svm.SVR(kernel=answer)
    except:
        messagebox.showinfo("Error", "Wrong kernel is selected")
        return ""
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""

    
    
    
    
    
    
# Machine Learning Functions
def XGB_regression(filename):
    model=xgb.XGBRegressor(objective="reg:linear")
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test,preds))
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    

    

    
    
def KNeighbour(filename):
    ans=simpledialog.askinteger("K Nearest Neighbours", "Number of Neighbours",
                                 parent=root,
                                 minvalue=1, maxvalue=1000)
    model=KNeighborsClassifier(n_neighbors = ans)
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    




def KMean(filename):
    ans=simpledialog.askinteger("K Mean Clustering", "Number of Clusters",
                                 parent=root,
                                 minvalue=1, maxvalue=1000)
    model=KMeans(n_clusters=ans)
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    

    
def DecisionTreeClas(filename):
    model=DecisionTreeClassifier()
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    
    
    
    
def NaiveBayesClas(filename):
    model=GaussianNB()
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    
    
    
    
def SVMClas(filename):
    answer = simpledialog.askstring("SVM", "Kernel Type(linear/poly/rbf/sigmoid)",
                                parent=app)
    try:
        model=svm.SVC(kernel=answer)
    except:
        messagebox.showinfo("Error", "Wrong kernel is selected")
        return ""
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
        
        
        
        

    
    
def XGBClas(filename):
    model=xgb.XGBClassifier(objective="binary:logistic")
    try:
        data = pd.read_csv(filename)
        d=[]
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d)-1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test,preds))
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""
    


    
#Frame for different models
f3=Frame(app)




#label for result
l1=Label(f3,text="Result:")
l1.grid(row=0,column=0)




# text field 
vark=StringVar()
textofresult=Entry(f3,textvariable=vark)
textofresult.grid(row=0,column=1)

    



#end
app.mainloop()