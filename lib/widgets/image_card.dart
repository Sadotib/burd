// lib/widgets/image_card.dart
import 'package:flutter/material.dart';
import 'dart:io';


class ImageCard extends StatelessWidget {
  final File? filePath;

  const ImageCard({super.key, required this.filePath});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.white,
      surfaceTintColor: Color(0xFF1F4BEA),
      clipBehavior: Clip.hardEdge,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        // side: BorderSide(color: Colors.pink),
      ),
      child: SizedBox(
        width: MediaQuery.of(context).size.width * 0.5,
        child: Column(
          children: [
            const SizedBox(height: 18),
            if (filePath != null)
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Image.file(filePath!, fit: BoxFit.contain),
              ),
            const SizedBox(height: 12),
            // Padding(
            //   padding: const EdgeInsets.all(8.0),
            //   child: Column(
            //     children: [
            //       Text("Prediction: ${BirdResult().prediction}"),
            //       Text(
            //         "Confidence: ${BirdResult().confidence.toStringAsFixed(2)}%",
            //       ),
            //     ],
            //   ),
            // ),
          ],
        ),
      ),
    );
  }
}
