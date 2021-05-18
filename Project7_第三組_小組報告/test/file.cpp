#include <string>
#include <vector>
#include <iostream>


class File {
public:
  File(std::string name) {
    this->_name = name;
  }

  File(File const & file) {
    this->_name = file._name;
  }

  std::string getName() const {
    return this->_name;
  }

private:
  std::string _name;
};


class Folder {
public:
  Folder(std::string name) {
    this->_name = name;
  }

  Folder(Folder const & folder) {
    this->copyFolder(folder);
  }
  
  Folder& operator= (Folder const & folder) {
    if (this == &folder) {
      return *this;
    }
    if (this->_files.size() > 0) {
      for (File * file : this->_files) {
        delete file;
      }
    }
    this->copyFolder(folder);
    return *this;
  }

  std::string getName() const {
    return this->_name;
  }

  void addNewFile(File * file) {
    this->_files.push_back(file);
  }

  File * getFile(int index) const {
    std::cout << this->_files.size() << std::endl;
      return this->_files.at(index);
  }

private:
  void copyFolder(Folder const & folder) {
    this->_name = folder._name;

    for (File * file : folder._files) {
      File * newFile = new File(*(file));
      this->_files.push_back(newFile);
    }
  }

private:
  std::string _name;
  std::vector<File *> _files;
};


void testFolder() {
  Folder * src = new Folder("src");

  std::cout << "First Folder's name: " << src->getName() << std::endl;
  Folder * bin = new Folder("bin");

  std::cout << "Second folder's name: " << bin->getName() << std::endl;

  bin = src;

  std::cout << "Second folder's new name is: " << bin->getName() << std::endl;

  bin->addNewFile(new File("main.cpp"));

  bin = bin;



  std::cout << "The folder src is still src, src's file is: " << bin->getFile(0)->getName() << std::endl;
}


int main(int argc, char const *argv[]) {
  testFolder();

  return 0;
}
