#include<stdio.h>
#include<stdlib.h>
#include<string.h>
//bir �st directory'ye ��kan komutu 'chdir ..' olarak de�il, 'chdir..' olarak, arada bo�luk b�rakmadan, giriniz. 
typedef struct node{
	char name[30];
	struct node *parent;
	struct node *firstchild;
	struct node *sibling;
}node;

void mkdir(char *, node *);
node* chdir(char *, node *);
node* chdirBefore(node *);
void rmdir(char *, node *);
void count(node *);
void dir(node *);

int main(){
	
	char komut[10], dosyaadi[30], komutsatiri[40];
	int kontrol=1;
	node *root,*curr;
	root=(node*)malloc(sizeof(node));
	strcpy(root->name,"Root Directory");
	root->parent=NULL;
	root->firstchild=NULL;
	root->sibling=NULL;
	curr=root;
	printf("Cikis icin q giriniz.\n");
	while(kontrol==1){
		printf(">");
		scanf(" %[^\n]s",komutsatiri);
		sscanf(komutsatiri,"%s%s",komut,dosyaadi);
		if(strcmp(komut,"mkdir")==0){
			mkdir(dosyaadi,curr);
		}else if(strcmp(komut,"chdir")==0){
			curr=chdir(dosyaadi,curr);
		}else if(strcmp(komut,"chdir..")==0 || strcmp(komut,"chdir ..")==0){
			curr=chdirBefore(curr);
		}else if(strcmp(komut,"rmdir")==0){
			rmdir(dosyaadi,curr);
		}else if(strcmp(komut,"dir")==0){
			dir(curr);
		}else if(strcmp(komut,"count")==0){
			count(curr);
		}else if(strcmp(komut,"q")==0){
			kontrol=-1;
		}else{
			printf("Hatali komut girdiniz.Lutfen tekrar deneyiniz.\n");
		}
	}
	
	
	return 0;
}

void mkdir(char name[], node *current){ 
	
	int kontrol;
	node *newnode, *iter;
	newnode=(node*)malloc(sizeof(node));
	strcpy(newnode->name,name); 
	newnode->parent=NULL;
	newnode->firstchild=NULL;
	newnode->sibling=NULL;
	
	iter=current->firstchild;
	if(current->firstchild!=NULL){
		if(strcmp(current->firstchild->name,name)!=0){
			while(iter->sibling!=NULL && strcmp(iter->sibling->name,name)!=0){
				iter=iter->sibling;
			}
			if(iter->sibling==NULL){
				kontrol=1; //ba�ta eleman var ve karde� elemanlar da var ve isimler farkl�ysa eklemesine izin ver
			}else{
				kontrol=-1; //ba�ta eleman var ve karde� elemanlar da var ama birinin ismi ayn�ysa eklemesine izin verme
			}
		}else{
			kontrol=-1; //ba�ta eleman varsa ama ad� ayn�ysa eklemesine izin verme
		}
	}else{
		kontrol=1; //ba�ta eleman yoksa eklemesine izin ver
	}
	
	if(kontrol==1){
		if(current->firstchild==NULL){
			current->firstchild=newnode;
			newnode->parent=current;
		}else{
			iter=current->firstchild;
			while(iter->sibling!=NULL){
				iter=iter->sibling;
			}
			iter->sibling=newnode;
			newnode->parent=current;
		}
		return;
		//ekleme yap�ld� ��nk� ayn� isimde ba�ka dosya yok
	}else{
		printf("Ayni isimli bir dosya zaten mevcut.\nLutfen farkli bir dosya adi giriniz.\n");
		 //ekleme yap�lamad� ��nk� ayn� isimde ba�ka dosya var
	}
	
}

node* chdir(char name[], node *current){ 
	node *iter;
	if(strcmp(current->firstchild->name,name)==0){
		current=current->firstchild;
	}else{
		iter=current->firstchild;
		while(iter->sibling!=NULL && strcmp(iter->sibling->name,name)!=0){
			iter=iter->sibling;
		}
		if(iter->sibling==NULL){
			printf("Boyle bir directory bulunmamakta.\n");
		}else{
			current=iter->sibling;
		}
	}
	return current;
}

node* chdirBefore(node *current){
	if(current->parent!=NULL){
		current=current->parent;
	}else{
		printf("Bir ust directory bulunmamakta.\nZaten Root Directory'desiniz.\n");
	}
	return current;
}

void rmdir(char name[], node *current){
	node *iter;
	iter=current->firstchild;
	
	
	if(iter==NULL){ //fc yok. fc:firstchild
		printf("Silinecek dosya bulunmamakta.\n");
		return;
	}else if(strcmp(current->firstchild->name,name)==0){ //fc var ve ad� ayn� 
		if(current->firstchild->firstchild!=NULL){ //fc'nin fc'si var m� diye kontrol ediyorum
			printf("Once altindaki dosyalari silmelisiniz.\n");
			return;
		}else{
			current->firstchild=current->firstchild->sibling; //fc'nin fc'si yoksa siliyorum.
		}
	}else if(iter->sibling==NULL){ //fc var ama ad� farkl� o y�zden sibling var m� diye soruyorum
		printf("Silinecek dosya bulunmamakta.\n");
		return;
	}else{
		while(iter->sibling!=NULL && strcmp(iter->sibling->name,name)!=0){ //sibling varsa silinecek siblingi bulana kadar gidiyorum
			iter=iter->sibling;
		}
		if(iter->sibling==NULL){  //siblingler aras�nda silinecek sibling var m� diye kontrol ediyorum
			printf("Silinecek dosya bulunmamakta.\n");
			return;
		}else if(iter->sibling->firstchild!=NULL){ //silinecek sibling bulunduysa alt�nda ba�ka directory'ler var m� diye kontrol ediyorum
			printf("Once altindaki dosyalari silmelisiniz.\n");
			return;
		}else{
			iter->sibling=iter->sibling->sibling; //silinecek sibling'in alt� bo�sa siliyorum
		}
	}
	 
	
}

void dir(node *current){
	
	node *iter;
	iter=current->firstchild;
	if(current->firstchild!=NULL){
		if(iter->sibling!=NULL){
			while(iter!=NULL){
				printf("%s\n",iter->name);
				iter=iter->sibling;
			}
		}else{
			printf("%s\n",current->firstchild->name);
		}
	}else{
		printf("Gosterilecek directory bulunmamakta.\n");
		return;
	}
	
}

void count(node *current){
	
	int sayac=0;
	node *iter;
	iter=current->firstchild;
	if(current->firstchild!=NULL){
		if(iter->sibling!=NULL){
			while(iter!=NULL){
				sayac++;
				iter=iter->sibling;
			}
			printf("%d\n",sayac);
		}else{
			printf("1\n");
		}
	}else{
		printf("0\n");
	}
	
}


