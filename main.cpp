#include <iostream>
#include <array>
#include <vector>
#include <queue>
#include <ctime>
#include <unistd.h>

using namespace std;

class Cell {
    //0 - Space; 1 - Wall; 2 - Food; 3 - Animal
    public:
    uint16_t type;
    uint32_t value;
};

class Animal {
    public:
    int32_t energy;
    float learning;
    
    int64_t id;
    int32_t xcord;
    int32_t ycord;

    uint32_t length;
    uint32_t width;

    //Gencode
    // Вокруг бота 8 клеток + 1 нейрон смещения
    array <float, 9> input;
    // Я решил, что должно быть 9 внутренних нейронов + 1 нейрон смещения
    array <float, 10> hidden;
    // 8 клеток вокруг бота, куда он мог пойти + 1 просто стоять
    array <float, 9> output;

    array <float, 90> inhid;
    array <float, 90> hidout;

    Animal(uint64_t _id, int32_t _xcord, int32_t _ycord, uint32_t _length, uint32_t _width) {
        id = _id;
        length = _length;
        width = _width;
        xcord = _xcord;
        ycord = _ycord;
        // Пусть начальная энергия - 200. Потолок - 1000
        energy = 100;
        learning = 0.08;
        // Добавил нейрон смещения
        input[input.size()-1] = 1;
        hidden[hidden.size()-1] = 1; 

        for(int i=0; i < inhid.size(); i++) {
            inhid[i] = (float)rand()/RAND_MAX;
        }
        for(int i=0; i < hidout.size(); i++) {
            hidout[i] = (float)rand()/RAND_MAX;
        }
    }
    Animal() {
        Animal(999L, 999, 999, 999, 999);
    }
    /*Animal copy() {
        return (*this);
    }*/
    void print_gencode() {
        for(int i=0; i < inhid.size(); i++) {
            cout << inhid[i] << "\t";
        }
        cout << "\n\n";
        for(int i=0; i < hidout.size(); i++) {
            cout << hidout[i] << "\t";
        }
        cout << endl;
    }
    Animal get_mutated(uint64_t _id, int32_t _xcord, int32_t _ycord, uint32_t _length, uint32_t _width) {

        Animal mutated(_id, _xcord, _ycord, length, width);
        mutated.inhid = inhid;
        mutated.hidout = hidout;
        for(int i=0; i < mutated.inhid.size(); i++) {
            // (1 - rand()%3) - выдает либо 1, либо 0 (то есть без изменений), либо -1, то есть уменьшаем вес.
            mutated.inhid[i] += (1 - rand()%3) * learning*(float)rand()/RAND_MAX;

        }
        for(int i=0; i < mutated.hidout.size(); i++) {
            // (1 - rand()%3) - выдает либо 1, либо 0 (то есть без изменений), либо -1, то есть уменьшаем вес.
            mutated.hidout[i] += (1 - rand()%3) * learning*(float)rand()/RAND_MAX;
        }

        return mutated;
    }
    // direction 0 - north; 1 - north-west; 2 - W; 3 - SW; 4 - S; 5 - SE; 6 - E; 7 - NE;
    void move(uint16_t direction) {
        switch(direction) {
            case 0: ycord+=1; break;
            case 1: ycord+=1; xcord-=1; break;
            case 2: xcord-=1; break;
            case 3: ycord-=1; xcord-=1; break;
            case 4: ycord-=1; break;
            case 5: ycord-=1; xcord+=1; break;
            case 6: xcord+=1; break;
            case 7: ycord+=1; xcord+=1; break;
            case 8: /*Don't move*/; break;
            default: cerr << "Bad 'direction'\n" << direction << ", but must be 0-7\n";
        }
        ycord = (ycord + width) % width;
        xcord = (xcord + length) % length;
        return;
    }
    void next_step(array <uint16_t, 8> surround, uint16_t print_choose) {
        energy -=10;

        if (surround.size() != 8) {
            cerr << "Bad array 'surround'\nSize: " << surround.size() << ", but must be 8\n";
        }
        // Копирую surround в input. input[input.size()-1] уже равен 1. Это нейрон смещения
        for(int i=0; i < surround.size(); i++) {
            input[i] = (float)surround[i];
            if(print_choose) cout << surround[i] << endl;
        }
        input[8] = 1;
        // Обнуляю нейроны hidden и output
        for(int i=0; i < hidden.size()-1; i++) {
            hidden[i] = 0;
        }
        for(int i=0; i < output.size(); i++) {
            output[i] = 0;
        }
        // Нахожу значения нейронов hidden
        for (int i=0; i < inhid.size(); i++) {
            hidden[i/input.size()] += inhid[i]*input[i%hidden.size()];
        }
        // Прогоняю результат через функцию активации LRELU
        for(int i=0; i < hidden.size(); i++) {
            hidden[i] = LRELU(hidden[i]);
        }
        // Теперь нахожу нейроны output
        for (int i=0; i < hidout.size(); i++) {
            output[i/hidden.size()] += hidout[i]*hidden[i%output.size()];
        }
        // Прогоняю результат через функцию активации LRELU
        for(int i=0; i < output.size(); i++) {
            output[i] = LRELU(output[i]);
        }
        // Ищу наиболее лучше направление (или стоять)
        uint16_t best_dir = 0;
        for(int i=0; i < output.size(); i++) {

            if (print_choose) cout << output[i] << endl;
            if(output[best_dir] < output[i]) {
                best_dir = i;
            }
        }
        move(best_dir);
    }
    void next_step(array <uint16_t, 8> surround) {
        next_step(surround, 0);
    }
    float LRELU(float x) {
        if(x < 0) return 0.01*x;
        else if(x >= 1) return 0.99 + 0.01*x;
        else return x;
    }
    ~Animal() {
        //cout << id << " был удален.\n";
    }
};
class Box {
    public:
    uint32_t length;
    uint32_t width;

    vector <Cell> field;
    uint32_t count_animals;
    vector <Animal> animals;
    queue <uint32_t> animals_ids;
    uint32_t count_alive_animals = 0;
    uint32_t count_food = 0;
    uint32_t food_value = 0;
    Box(uint32_t _length, uint32_t _width, uint32_t _count_animals, uint32_t _count_food, uint32_t _food_value) {
        length = _length;
        width = _width;

        field.resize(length*width);
        count_animals = _count_animals;
        create_animals();
        count_food = _count_food;
        food_value = _food_value; 
        create_food();
    }
    void create_animals() {
        for(uint64_t i = 0; i < count_animals; i++) {
            animals.resize(count_animals);

            int32_t xcord;
            int32_t ycord;
            do {
                xcord = rand()%length;
                ycord = rand()%width;
            } while(field[ycord*length + xcord].type != 0);
            Animal new_animal(i, xcord, ycord, length, width);
            animals[i] = new_animal;
            field[ycord*length + xcord].type = 3;
            field[ycord*length + xcord].value = i;
            animals_ids.push(i);

        }
    }
    void create_food() {
        for(int i=0; i < count_food; i++) {
            animals.resize(count_food);

            int32_t xcord;
            int32_t ycord;
            do {
                xcord = rand()%length;
                ycord = rand()%width;
            } while(field[ycord*length + xcord].type != 0);
            
            field[ycord*length + xcord].type = 2;
            field[ycord*length + xcord].value = food_value;
        }
    }
    array <uint16_t, 8> get_surround(int32_t _xcord, int32_t _ycord) {
        array <uint16_t, 8> surround;

        // Do you know more perfect formul for this 16 string? Come on! Code it! 
        _ycord = (_ycord+1+width)%width;
        surround[0] = field[(_ycord)*length + _xcord].type;
        _xcord = (_xcord-1+length)%length;
        surround[1] = field[(_ycord)*length + _xcord].type;
        _ycord = (_ycord-1+width)%width;
        surround[2] = field[(_ycord)*length + _xcord].type;
        _ycord = (_ycord-1+width)%width;
        surround[3] = field[(_ycord)*length + _xcord].type;
        _xcord = (_ycord+1+length)%length;
        surround[4] = field[(_ycord)*length + _xcord].type;
        _xcord = (_ycord+1+length)%length;
        surround[5] = field[(_ycord)*length + _xcord].type;
        _ycord = (_ycord+1+width)%width;
        surround[6] = field[(_ycord)*length + _xcord].type;
        _ycord = (_ycord+1+width)%width;
        surround[7] = field[(_ycord)*length + _xcord].type;

        return surround;
    } 
    void draw() {
        for(int y=0; y < width; y++) {
            for(int x=0; x < length; x++) {
                switch(field[length*y + x].type) {
                    case 0: cout << '.'; break;
                    case 1: cout << '#'; break;
                    case 2: cout << '&'; break;
                    case 3: cout << '@'; break;
                }
            }
            cout << endl;
        }
        cout << "--##--##--##--##--##--##--##--\n";
    }
    void draw_live(uint32_t seconds) {
        draw();
        sleep(seconds);
    }
    void next_epoch() {
        for(int i=0; i < field.size(); i++) {
            field[i].type = 0;
        }
        vector <Animal> survived_animals(10);
        for(int i=0; i < 10; i++) {
            survived_animals[i] = animals[animals_ids.front()];
            animals_ids.pop();
        }
        for(int i=0; i < 10; i++) {
            animals[i] = survived_animals[i];
            animals[i].energy = 100;
            animals[i].id = i;
            int32_t xcord;
            int32_t ycord;
            do {
                xcord = rand()%length;
                ycord = rand()%width;
            } while(field[ycord*length + xcord].type != 0);
            animals[i].xcord = xcord;
            animals[i].ycord = ycord;

            field[ycord*length + xcord].type = 3;
            field[ycord*length + xcord].value = i;
            animals_ids.push(i);
        }
        for(uint64_t i = 10; i < count_animals; i++) {
            int32_t xcord;
            int32_t ycord;
            do {
                xcord = rand()%length;
                ycord = rand()%width;
            } while(field[ycord*length + xcord].type != 0);
            Animal new_animal = animals[i/10].get_mutated(i, xcord, ycord, length, width);
            animals[i] = new_animal;
            field[ycord*length + xcord].type = 3;
            field[ycord*length + xcord].value = i;
            animals_ids.push(i);
        }
        create_food();
    }
    void run(uint32_t iterations, uint16_t withdraw, uint32_t seconds) {
        for(int i=0; i < iterations;) {
            if(withdraw) draw_live(seconds);
            if(animals_ids.size() <= 10) {
                i++;
                next_epoch();
            }  
            uint32_t current_id = animals_ids.front();
            animals_ids.pop();
            //cout << current_id << endl;
            if(animals[current_id].energy <= 0) {
                //cout << "Кекнулся\n";
                continue;
            }
            int32_t xcord = animals[current_id].xcord;
            int32_t ycord = animals[current_id].ycord;
            animals[current_id].next_step(get_surround(xcord, ycord));
            field[length*ycord + xcord].type = 0;
            xcord = animals[current_id].xcord;
            ycord = animals[current_id].ycord;
            uint16_t type = field[length*ycord + xcord].type;
            uint32_t value = field[length*ycord + xcord].value;
            switch(type) {
                // Now there is animal
                case 0:
                    field[length*ycord + xcord].type = 3;
                    field[length*ycord + xcord].value = current_id;
                    animals_ids.push(current_id);
                    break;
                // Wall(?)
                case 1:
                    cerr << "I didn't add walls...\n"; break;
                // Нашел Food
                case 2: 
                    //cout << "Нашел еду\n";
                    animals[current_id].energy+= value;
                    field[length*ycord + xcord].type = 3;
                    field[length*ycord + xcord].value = current_id;
                    animals_ids.push(current_id);
                    break;
                // Столкнулся с Animal
                case 3:
                    if (animals[value].energy > animals[current_id].energy) {
                        animals[value].energy -= animals[current_id].energy;
                        animals[current_id].energy = -1;
                    }
                    else if(animals[value].energy == animals[current_id].energy) {
                        animals[value].energy = 1;
                        animals[current_id].energy = -1;
                    }
                    else {
                        animals[current_id].energy -= animals[value].energy;
                        animals[value].energy = -1;
                        field[length*ycord + xcord].type = 3;
                        field[length*ycord + xcord].value = current_id;
                        animals_ids.push(current_id);
                    }
                    break;
                // Next iteration!
            }
        }
    }

};
int main() {
    time_t t;
    srand(time(&t));
    uint32_t length, width, count_animals, count_food, food_value, seconds; 
    cin >> length >> width >> count_animals >> count_food >> food_value >> seconds;
    // 9, 9, 12, 12, 250
    Box my_box(length, width, count_animals, count_food, food_value);
    my_box.run(10, 1, seconds);
    /*
    //my_box.run(30, 1);
    for(int i=0; i < 10; i++) {
        my_box.animals[i].print_gencode();
        my_box.animals[i].next_step(my_box.get_surround(my_box.animals[i].xcord, my_box.animals[i].ycord), 1);
    }
    my_box.run(1000, 0, 1);
    
    for(int i=0; i < 10; i++) {
        my_box.animals[i].print_gencode();
        my_box.animals[i].next_step(my_box.get_surround(my_box.animals[i].xcord, my_box.animals[i].ycord), 1);
    }
    for(int i=0; i< 1000000000; i++);
    my_box.run(2, 1);

    */
    return 0;
}