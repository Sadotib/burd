

class LabelImageHelper {
  static Future<Map<String, String>> getLabelImageMap() async {
    
    // final labels = labelsText
    //     .split('\n')
    //     .map((e) => e.trim())
    //     .where((e) => e.isNotEmpty);

    final Map<String, String> labelImageMap = {
      "Asian Green Bee Eater":
          "https://i.postimg.cc/Kjb0L6hT/Asian-Green-Bee-Eater.jpg",
      "Brown Headed Barbet":
          "https://i.postimg.cc/SRGDGgmv/Brown-Headed-Barbet.jpg",
      "Cattle Egret": "https://i.postimg.cc/sfP6yg54/Cattle-Egret.jpg",
      "Common Kingfisher":
          "https://i.postimg.cc/5NxPtW3d/Common-Kingfisher.jpg",
      "Common Myna": "https://i.postimg.cc/jSD4Wxn9/Common-Myna.jpg",
      "Common Rosefinch": "https://i.postimg.cc/j28y5mxt/Common-Rosefinch.jpg",
      "Common Tailorbird":
          "https://i.postimg.cc/FHk0rT2X/Common-Tailorbird.jpg",
      "Coppersmith Barbet":
          "https://i.postimg.cc/Fs8JBj1W/Coppersmith-Barbet.jpg",
      "Forest Wagtail": "https://i.postimg.cc/xTJXMcgK/Forest-Wagtail.jpg",
      "Gray Wagtail": "https://i.postimg.cc/ZR5RRRCv/Gray-Wagtail.jpg",
      "Hoopoe": "https://i.postimg.cc/DfDZv7Fm/Hoopoe.jpg",
      "House Crow": "https://i.postimg.cc/xdtfW2tf/House-Crow.jpg",
      "Indian Grey Hornbill":
          "https://i.postimg.cc/wvMqzGGh/Indian-Grey-Hornbill.jpg",
      "Indian Peacock": "https://i.postimg.cc/dDPw8ttB/Indian-Peacock.jpg",
      "Indian Pitta": "https://i.postimg.cc/zD9gdCyd/Indian-Pitta.jpg",
      "Indian Roller": "https://i.postimg.cc/nhxQxnYB/Indian-Roller.jpg",
      "Jungle Babbler": "https://i.postimg.cc/q7MhtQF7/Jungle-Babbler.jpg",
      "Northern Lapwing": "https://i.postimg.cc/YSn4QYcz/Northern-Lapwing.jpg",
      "Red Wattled Lapwing":
          "https://i.postimg.cc/xCBc4Q1k/Red-Wattled-Lapwing.jpg",
      "Ruddy Shelduck": "https://i.postimg.cc/KcgKg10t/Ruddy-Shelduck.jpg",
      "Rufous Treepie": "https://i.postimg.cc/HsDr0TNR/Rufous-Treepie.jpg",
      "Sarus Crane": "https://i.postimg.cc/pXfpqgbT/Sarus-Crane.jpg",
      "White Breasted Kingfisher":
          "https://i.postimg.cc/mgThq8BS/White-Breasted-Kingfisher.jpg",
      "White Breasted Waterhen":
          "https://i.postimg.cc/1ztXRXgn/White-Breasted-Waterhen.jpg",
      "White Wagtail": "https://i.postimg.cc/Kz3YLWDm/White-Wagtail.jpg",
    };
    // for (final label in labels) {
    //   labelImageMap[label] = label; // Assumes fixed naming
    // }
    return labelImageMap;
  }
}
